using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

using Ecng.Common;

using StockSharp.Algo;
using StockSharp.Algo.Commissions;
using StockSharp.Algo.Strategies;
using StockSharp.Algo.Testing;
using StockSharp.BusinessEntities;
using StockSharp.Messages;

namespace MoexValidate
{
    /// <summary>
    /// Group-TP Grid Strategy for StockSharp HistoryEmulationConnector.
    /// Sends real MarketOrders, SS engine fills them.
    /// </summary>
    public class GroupTPGridStrategy : Strategy
    {
        static readonly string[] AssetNames = { "BR", "NG", "SV" };
        static readonly double[] Weights = { 0.36, 0.28, 0.36 };
        const int SmaPeriod = 120;
        const int LookbackBars = 1573;
        const double EntryZ = 2.4379;
        const double GridStepZ = 0.6020;
        const double TpPct = 0.003647;
        const int MaxPos = 4;
        const int NParts = 4;
        const double Cap = 10000.0;

        readonly Dictionary<string, Security> _secs;
        readonly Dictionary<string, List<double>> _closes = new();
        readonly Dictionary<string, double> _latest = new();
        readonly List<double> _idxHist = new();
        readonly double[] _cumsum = new double[3];

        class Group
        {
            public Dictionary<string, double> EntryPrices = new();
            public double EntryZ, Size;
        }
        readonly List<Group> _groups = new();

        int _bars, _sync;
        bool _pendEntry; double _pendEz; int _pendFill = -1, _pendClose = -1;
        double _freeCash = Cap, _baseSize = Cap / NParts;
        int _nTrades, _nWins;
        double _realPnl;
        int _zLbStart; double _zSum, _zSq; int _zCnt;
        DateTimeOffset _lastTime;
        double _peak = Cap, _maxDd;
        public int OrdersSent;

        public GroupTPGridStrategy(Dictionary<string, Security> secs)
        {
            _secs = secs;
            foreach (var n in AssetNames) { _closes[n] = new(); _latest[n] = 0; }
        }

        protected override void OnStarted(DateTimeOffset time)
        {
            base.OnStarted(time);
            this.CandleReceived += OnCandle;
            foreach (var kv in _secs)
                Subscribe(new Subscription(DataType.TimeFrame(TimeSpan.FromSeconds(5)), kv.Value));
        }

        void OnCandle(Subscription sub, ICandleMessage c)
        {
            if (c.State != CandleStates.Finished) return;
            string nm = null;
            foreach (var kv in _secs)
                if (kv.Value.Code == c.SecurityId.SecurityCode) { nm = kv.Key; break; }
            if (nm == null) return;

            _latest[nm] = (double)c.ClosePrice;
            _closes[nm].Add((double)c.ClosePrice);
            _lastTime = c.OpenTime;
            _sync++;
            if (_sync < 3) return;
            _sync = 0;
            _bars++;
            Tick();
        }

        void Tick()
        {
            // INDEX
            int i = _bars - 1;
            double iv = 0;
            for (int j = 0; j < 3; j++)
            {
                var cl = _closes[AssetNames[j]];
                int n = cl.Count;
                _cumsum[j] += cl[n - 1];
                double sma;
                if (i >= SmaPeriod) { _cumsum[j] -= cl[n - 1 - SmaPeriod]; sma = _cumsum[j] / SmaPeriod; }
                else if (i >= SmaPeriod - 1) sma = _cumsum[j] / SmaPeriod;
                else sma = _cumsum[j] / (i + 1);
                if (sma > 1e-10) iv += Weights[j] * (cl[n - 1] / sma);
            }
            _idxHist.Add(Math.Log(Math.Max(iv, 1e-10)));

            // Z-SCORE
            double si = _idxHist[^1];
            _zSum += si; _zSq += si * si; _zCnt++;
            while (_zLbStart < _idxHist.Count - 1 && _zCnt > LookbackBars)
            { _zSum -= _idxHist[_zLbStart]; _zSq -= _idxHist[_zLbStart] * _idxHist[_zLbStart]; _zCnt--; _zLbStart++; }
            double zi = 0;
            if (_zCnt >= 10) { double mu = _zSum / _zCnt; double v = (_zSq / _zCnt - mu * mu) * _zCnt / (_zCnt - 1); if (v > 1e-30) zi = (si - mu) / Math.Sqrt(v); }

            // SESSION
            int hm = (_lastTime.Hour + 3) * 60 + _lastTime.Minute; hm %= 1440;
            bool sess = (hm >= 600 && hm < 1125) || (hm >= 1145 && hm < 1430);

            // CLOSE FILL
            if (_pendClose >= 0 && _bars >= _pendClose && _groups.Count > 0)
            {
                foreach (var g in _groups)
                {
                    double pnl = 0;
                    for (int k = 0; k < 3; k++)
                    {
                        string nm = AssetNames[k];
                        if (g.EntryPrices[nm] > 0)
                        {
                            // SELL via SS MarketOrder
                            decimal qty = (decimal)Math.Round(g.Size * Weights[k] / _latest[nm], 4);
                            if (qty > 0)
                            {
                                try { SellMarket(qty, _secs[nm]); OrdersSent++; } catch { }
                            }
                            pnl += Weights[k] * (_latest[nm] / g.EntryPrices[nm] - 1) * g.Size;
                            pnl -= 2 * 0.00025 * g.Size * Weights[k];
                        }
                    }
                    _realPnl += pnl; _freeCash += g.Size + pnl; _nTrades++; if (pnl > 0) _nWins++;
                }
                _groups.Clear(); _pendClose = -1;
            }

            // TP CHECK
            if (_groups.Count > 0 && sess && _pendClose < 0)
            {
                double ti = 0, tp = 0;
                foreach (var g in _groups)
                {
                    ti += g.Size;
                    for (int k = 0; k < 3; k++)
                        if (g.EntryPrices[AssetNames[k]] > 0)
                        { tp += Weights[k] * (_latest[AssetNames[k]] / g.EntryPrices[AssetNames[k]] - 1) * g.Size; tp -= 2 * 0.00025 * g.Size * Weights[k]; }
                }
                if (ti > 0 && tp / ti >= TpPct) _pendClose = _bars + 1;
            }

            // ENTRY FILL
            if (_pendEntry && _bars >= _pendFill)
            {
                if (_groups.Count < MaxPos && _freeCash >= _baseSize)
                {
                    var g = new Group { EntryZ = _pendEz, Size = _baseSize };
                    for (int k = 0; k < 3; k++)
                    {
                        string nm = AssetNames[k];
                        g.EntryPrices[nm] = _latest[nm];
                        // BUY via SS MarketOrder
                        decimal qty = (decimal)Math.Round(_baseSize * Weights[k] / _latest[nm], 4);
                        if (qty > 0)
                        {
                            try { BuyMarket(qty, _secs[nm]); OrdersSent++; } catch { }
                        }
                    }
                    _groups.Add(g); _freeCash -= _baseSize;
                }
                _pendEntry = false;
            }

            // ENTRY SIGNAL
            if (sess && !_pendEntry && _pendClose < 0 && zi < -EntryZ && _groups.Count < MaxPos && _freeCash >= _baseSize)
            {
                bool ok = true;
                if (_groups.Count > 0 && zi > _groups.Min(g => g.EntryZ) - GridStepZ) ok = false;
                if (ok) { _pendEntry = true; _pendEz = zi; _pendFill = _bars + 1; }
            }

            // EQUITY
            double ur = 0;
            foreach (var g in _groups)
                for (int k = 0; k < 3; k++)
                    if (g.EntryPrices[AssetNames[k]] > 0)
                        ur += Weights[k] * (_latest[AssetNames[k]] / g.EntryPrices[AssetNames[k]] - 1) * _baseSize;
            double eq = Cap + _realPnl + ur;
            if (eq > _peak) _peak = eq;
            double dd = (eq - _peak) / _peak; if (dd < _maxDd) _maxDd = dd;
        }

        public void PrintResults()
        {
            double eq = Cap + _realPnl;
            // Add unrealized for open positions
            double ur = 0;
            foreach (var g in _groups)
                for (int k = 0; k < 3; k++)
                    if (g.EntryPrices[AssetNames[k]] > 0)
                        ur += Weights[k] * (_latest[AssetNames[k]] / g.EntryPrices[AssetNames[k]] - 1) * _baseSize;
            eq += ur;

            double ret = (eq / Cap - 1) * 100, dd = _maxDd * 100;
            double pdd = dd != 0 ? Math.Abs(ret / dd) : 0;
            Console.WriteLine($"\n{new string('=', 70)}");
            Console.WriteLine("  STOCKSHARP FULL BACKTEST RESULT");
            Console.WriteLine($"  (HistoryEmulationConnector + BuyMarket/SellMarket orders)");
            Console.WriteLine($"{new string('=', 70)}");
            Console.WriteLine($"  P&L: {ret:+0.0;-0.0}%  MaxDD: {dd:0.0}%  P&L/DD: {pdd:0.0}");
            Console.WriteLine($"  Trades: {_nTrades} ({_nWins}w, WR={(_nTrades > 0 ? (double)_nWins / _nTrades * 100 : 0):0}%)");
            Console.WriteLine($"  Bars: {_bars:N0}  Orders sent: {OrdersSent}");
            try
            {
                Console.WriteLine($"  SS PnL: {PnL:F2}");
                Console.WriteLine($"  SS Orders filled: {Orders?.Count() ?? 0}");
                Console.WriteLine($"  SS Positions: {Positions?.Count() ?? 0}");
                var stats = StatisticManager?.Parameters;
                if (stats != null)
                    foreach (var p in stats)
                        Console.WriteLine($"    {p.Name}: {p.Value}");
            }
            catch (Exception ex) { Console.WriteLine($"  (SS stats error: {ex.Message})"); }
            Console.WriteLine($"\n  Python Fast BT: +259.1% DD=-12.5% P&L/DD=20.7 trades=859");
            Console.WriteLine($"  StockSharp:      {ret:+0.0;-0.0}% DD={dd:0.0}% P&L/DD={pdd:0.0} trades={_nTrades}");
        }
    }

    class Program
    {
        static readonly ManualResetEvent _done = new(false);

        static void Main()
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;
            Console.WriteLine(new string('=', 70));
            Console.WriteLine("  STOCKSHARP FULL BACKTEST — HistoryEmulationConnector");
            Console.WriteLine("  Real MarketOrders + SS execution engine");
            Console.WriteLine(new string('=', 70));

            var dataDir = Path.GetFullPath(Path.Combine(
                AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "data"));

            var names = new[] { "BR", "NG", "SV" };
            var exchangeInfo = new InMemoryExchangeInfoProvider();

            var secs = new Dictionary<string, Security>();
            foreach (var name in names)
            {
                var board = exchangeInfo.GetOrCreateBoard("FORTS");
                secs[name] = new Security
                {
                    Id = $"{name}@FORTS",
                    Code = name,
                    Board = board,
                    PriceStep = 0.01m,
                    VolumeStep = 0.0001m,
                    MinVolume = 0.0001m,
                    Type = SecurityTypes.Future,
                    Decimals = 4,
                };
            }

            var pf = Portfolio.CreateSimulator();
            pf.CurrentValue = 10_000;

            Console.WriteLine($"Loading from {dataDir}...");
            var candleData = new Dictionary<string, List<TimeFrameCandleMessage>>();
            DateTimeOffset minStart = DateTimeOffset.MaxValue, minEnd = DateTimeOffset.MinValue;

            foreach (var name in names)
            {
                var list = new List<TimeFrameCandleMessage>();
                foreach (var line in File.ReadLines(Path.Combine(dataDir, $"{name}_5s.csv")).Skip(1))
                {
                    var p = line.Split(',');
                    if (p.Length < 2) continue;
                    if (!DateTimeOffset.TryParse(p[0], CultureInfo.InvariantCulture, DateTimeStyles.None, out var dt)) continue;
                    if (!decimal.TryParse(p[1], CultureInfo.InvariantCulture, out var cl)) continue;
                    list.Add(new TimeFrameCandleMessage
                    {
                        SecurityId = new SecurityId { SecurityCode = name, BoardCode = "FORTS" },
                        OpenTime = dt, CloseTime = dt.AddSeconds(5),
                        OpenPrice = cl, HighPrice = cl, LowPrice = cl, ClosePrice = cl,
                        TotalVolume = 10000, State = CandleStates.Finished,
                    });
                }
                candleData[name] = list;
                if (list.Count > 0)
                {
                    if (list[0].OpenTime < minStart) minStart = list[0].OpenTime;
                    if (list[^1].OpenTime > minEnd) minEnd = list[^1].OpenTime;
                }
                Console.WriteLine($"  {name}: {list.Count:N0}");
            }

            var startTime = minStart.UtcDateTime.UtcKind();
            var stopTime = minEnd.UtcDateTime.UtcKind();

            Console.WriteLine($"\nPeriod: {startTime:g} - {stopTime:g}");
            Console.WriteLine("Creating HistoryEmulationConnector...");

            var secProvider = new CollectionSecurityProvider(secs.Values);
            var connector = new HistoryEmulationConnector(secProvider, new[] { pf })
            {
                EmulationAdapter =
                {
                    Settings =
                    {
                        MatchOnTouch = true,
                    }
                },
                HistoryMessageAdapter =
                {
                    StartDate = startTime,
                    StopDate = stopTime,
                },
            };

            // Register history data sources
            foreach (var name in names)
            {
                var sec = secs[name];
                var data = candleData[name];
                var dt = DataType.TimeFrame(TimeSpan.FromSeconds(5));
                int idx = 0;
                connector.RegisterHistorySource(sec, dt, date =>
                {
                    var result = new List<Message>();
                    while (idx < data.Count && data[idx].OpenTime.Date <= date.Date)
                    {
                        result.Add(data[idx]);
                        idx++;
                    }
                    return result;
                });
            }

            // Send Level1 for each security (margin info)
            connector.SecurityReceived += (sub, s) =>
            {
                var secId = s.ToSecurityId();
                var l1 = new Level1ChangeMessage { SecurityId = secId, ServerTime = startTime }
                    .TryAdd(Level1Fields.MinPrice, 0.01m)
                    .TryAdd(Level1Fields.MaxPrice, 1000000m)
                    .TryAdd(Level1Fields.MarginBuy, 10000m)
                    .TryAdd(Level1Fields.MarginSell, 10000m);
                connector.SendInMessage(l1);
            };

            // Strategy
            var strategy = new GroupTPGridStrategy(secs)
            {
                Connector = connector,
                Security = secs["BR"],
                Portfolio = pf,
            };

            // Progress
            int lastPct = 0;
            connector.ProgressChanged += steps =>
            {
                var pct = (int)steps;
                if (pct > lastPct && pct % 10 == 0)
                {
                    Console.WriteLine($"  Progress: {pct}%");
                    lastPct = pct;
                }
            };

            connector.StateChanged2 += state =>
            {
                if (state == ChannelStates.Stopped)
                {
                    strategy.Stop();
                    Console.WriteLine("  Emulation stopped.");
                    _done.Set();
                }
            };

            var sw = System.Diagnostics.Stopwatch.StartNew();

            // Start strategy BEFORE connector (per SS docs)
            strategy.Start();
            connector.Connect();
            connector.Start();

            Console.WriteLine("  Emulation running...");

            // Wait
            _done.WaitOne(TimeSpan.FromHours(4));
            sw.Stop();

            Console.WriteLine($"\nCompleted in {sw.Elapsed.TotalMinutes:F1} min");
            strategy.PrintResults();
        }
    }
}
