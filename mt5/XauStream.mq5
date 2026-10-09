//+------------------------------------------------------------------+
//| XauStream.mq5 — READ-ONLY market data streamer for xau-terminal. |
//| Sends bar history on start (and every minute), then ticks every  |
//| TimerMs. It contains no trading calls: it cannot open, modify or |
//| close positions.                                                 |
//|                                                                  |
//| Setup: Tools > Options > Expert Advisors > "Allow WebRequest for |
//| listed URL" -> add  http://127.0.0.1:8000                        |
//+------------------------------------------------------------------+
#property strict
#property description "xau-terminal read-only streamer (no trading)"

input string Endpoint    = "http://127.0.0.1:8000/api/ingest";
input string ApiKey      = "";           // must match STREAM_KEY on the backend (optional)
input string Suffix      = "";           // broker symbol suffix, e.g. ".r" or "m"
input int    TimerMs     = 250;          // tick push cadence
input int    HistoryBars = 600;          // M5/M15/H1/D1
input int    HistoryBarsM1 = 3000;       // ~2 trading days of M1

string Base[] = {"XAUUSD","XAUEUR","XAUGBP","XAUJPY","XAUCHF","XAUAUD",
                 "EURUSD","GBPUSD","USDJPY","USDCHF","AUDUSD"};
ENUM_TIMEFRAMES TFs[] = {PERIOD_M1, PERIOD_M5, PERIOD_M15, PERIOD_H1, PERIOD_D1};
string TFNames[]      = {"M1", "M5", "M15", "H1", "D1"};
long   LastMsc[];
datetime LastHistory = 0;

string Sym(int i) { return Base[i] + Suffix; }

bool Post(const string json, int timeoutMs = 3000)
{
   char data[], result[];
   string resultHeaders;
   int n = StringToCharArray(json, data, 0, WHOLE_ARRAY, CP_UTF8);
   ArrayResize(data, n - 1);                                   // drop trailing \0
   string headers = "Content-Type: application/json\r\n";
   if(ApiKey != "") headers += "X-Stream-Key: " + ApiKey + "\r\n";
   int code = WebRequest("POST", Endpoint, headers, timeoutMs, data, result, resultHeaders);
   if(code == -1) { Print("XauStream WebRequest error ", GetLastError(), " - is the URL allowed in Options?"); return false; }
   return code == 200;
}

string Offset() { return IntegerToString((long)(TimeTradeServer() - TimeGMT())); }  // server - UTC, seconds

// One small POST per symbol x timeframe (a single multi-MB JSON times out under Wine).
// full=true: HistoryBars; full=false: last 3 bars only, to resync closed bars cheaply.
void SendHistory(bool full)
{
   LastHistory = TimeCurrent();                                 // set first: a failure retries in a minute, not every tick
   for(int i = 0; i < ArraySize(Base); i++)
   {
      if(!SymbolSelect(Sym(i), true)) continue;                // broker may not quote every XAU cross
      int d = (int)SymbolInfoInteger(Sym(i), SYMBOL_DIGITS);
      for(int k = 0; k < ArraySize(TFs); k++)
      {
         MqlRates r[];
         ArraySetAsSeries(r, false);
         int want = full ? (TFs[k] == PERIOD_M1 ? HistoryBarsM1 : HistoryBars) : 3;
         int got = CopyRates(Sym(i), TFs[k], 0, want, r);
         if(got <= 0) continue;
         string bars[];
         ArrayResize(bars, got);
         for(int j = 0; j < got; j++)
            bars[j] = StringFormat("[%I64d,%s,%s,%s,%s,%I64d]", (long)r[j].time,
                      DoubleToString(r[j].open, d), DoubleToString(r[j].high, d), DoubleToString(r[j].low, d),
                      DoubleToString(r[j].close, d), r[j].tick_volume);
         string joined = "";
         for(int j = 0; j < got; j++) joined += (j ? "," : "") + bars[j];
         Post("{\"offset\":" + Offset() + ",\"history\":[{\"sym\":\"" + Base[i] + "\",\"tf\":\"" + TFNames[k]
              + "\",\"bars\":[" + joined + "]}]}", 15000);
      }
   }
}

int OnInit()
{
   ArrayResize(LastMsc, ArraySize(Base));
   ArrayInitialize(LastMsc, 0);
   SendHistory(true);
   EventSetMillisecondTimer(TimerMs);
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason) { EventKillTimer(); }

void OnTimer()
{
   if(TimeCurrent() - LastHistory >= 60) SendHistory(false);  // resync last closed bars once a minute
   string ticks = "";
   for(int i = 0; i < ArraySize(Base); i++)
   {
      MqlTick t;
      if(!SymbolInfoTick(Sym(i), t) || t.time_msc == LastMsc[i]) continue;
      LastMsc[i] = t.time_msc;
      int d = (int)SymbolInfoInteger(Sym(i), SYMBOL_DIGITS);
      ticks += (ticks == "" ? "" : ",") + StringFormat("{\"sym\":\"%s\",\"t\":%I64d,\"bid\":%s,\"ask\":%s}",
               Base[i], t.time_msc, DoubleToString(t.bid, d), DoubleToString(t.ask, d));
   }
   if(ticks != "") Post("{\"offset\":" + Offset() + ",\"ticks\":[" + ticks + "]}");
}
