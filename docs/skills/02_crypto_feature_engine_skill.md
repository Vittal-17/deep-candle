# Crypto Feature Engine Skill

## 1. Temporal Boundary & Anti-Leakage Invariants
- **Mathematical Proof of $t_0$ Observable Limit:**
  Let $T$ be the current wall-clock time. For a timeframe $\Delta t$, a bar is only observable if its closing time $t_{end} = t_{start} + \Delta t \le T$.
  - Example: At $T = \text{14:15:00 UTC}$, the `15m` candle spanning `14:00:00` to `14:14:59` is fully closed and valid.
  - The `1h` candle ending at `14:00:00` is valid. The ongoing `1h` candle spanning `14:00:00` to `14:59:59` is strictly inaccessible. The `1h` feature vector must use the lagged bar ending at `14:00:00`.
- **Warmup Data Requirements:**
  To compute all indicators without initialization artifacts, the engine requires:
  - Minimum 30 closed `15m` bars.
  - Minimum 220 closed `1h` bars.
  - Minimum 6 closed `5m` bars.

## 2. Mathematical Formulations for All 11 Features
Let $C_t, H_t, L_t, V_t$ denote Close, High, Low, and Volume at time $t$.

1. **`atr_pct_15m`**: 14-period Wilder's smoothed ATR divided by close price.
   $TR_t = \max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)$
   $ATR_{14,t} = \frac{13 \times ATR_{14,t-1} + TR_t}{14}$
   $\text{atr\_pct\_15m} = \frac{ATR_{14,t}}{C_t}$

2. **`bb_squeeze_15m`**: Bollinger Band width normalized by 20 SMA.
   $\mu_t = \frac{1}{20} \sum_{i=0}^{19} C_{t-i}$, $\sigma_t = \sqrt{\frac{1}{20} \sum_{i=0}^{19} (C_{t-i} - \mu_t)^2}$
   $\text{bb\_squeeze\_15m} = \frac{4 \times \sigma_t}{\mu_t}$

3. **`vol_zscore_15m`**: 20-period volume mean and standard deviation Z-score.
   $\mu_V = \frac{1}{20} \sum_{i=0}^{19} V_{t-i}$, $\sigma_V = \sqrt{\frac{1}{20} \sum_{i=0}^{19} (V_{t-i} - \mu_V)^2}$
   $\text{vol\_zscore\_15m} = \frac{V_t - \mu_V}{\sigma_V}$

4. **`vwap_dist_atr_15m`**: Intraday VWAP distance divided by 15m ATR. VWAP resets daily at 00:00:00 UTC.
   $VWAP_t = \frac{\sum_{j=0}^{t} (H_j + L_j + C_j)/3 \times V_j}{\sum_{j=0}^{t} V_j}$
   $\text{vwap\_dist\_atr\_15m} = \frac{C_t - VWAP_t}{ATR_{14,t}}$

5. **`rsi_15m`**: Standard 14-period Wilder's RSI.
   $U_t = \max(C_t - C_{t-1}, 0)$, $D_t = \max(C_{t-1} - C_t, 0)$
   $RS = \frac{\text{SMMA}_{14}(U)}{\text{SMMA}_{14}(D)}$
   $\text{rsi\_15m} = 100 - \frac{100}{1 + RS}$

6. **`rsi_1h`**: Lagged 1h 14-period RSI.
   Calculated same as above, but using the $1h$ timeframe series, lagged to the most recently closed $1h$ boundary.

7. **`ema_200_dist_1h`**: Distance between lagged 1h close and 1h 200 EMA, divided by 1h ATR.
   $EMA_{200,t} = C_t \times \frac{2}{201} + EMA_{200,t-1} \times (1 - \frac{2}{201})$
   $\text{ema\_200\_dist\_1h} = \frac{C_{1h} - EMA_{200,1h}}{ATR_{14,1h}}$

8. **`micro_momentum_ratio`**: Return of latest 5m candle divided by full 15m candle return.
   $\text{micro\_momentum\_ratio} = \frac{(C_{5m} - O_{5m}) / O_{5m}}{((C_{15m} - O_{15m}) / O_{15m}) + 1e-7}$

9. **`micro_volume_acceleration`**: Volume of latest 5m candle divided by average volume of previous two 5m candles.
   $\text{micro\_volume\_acceleration} = \frac{V_{5m,t}}{((V_{5m,t-1} + V_{5m,t-2}) / 2) + 1e-7}$

10. **`btc_relative_delta`**: 15m return of target coin minus 15m return of BTCUSDT.
    $R_{\text{target}} = \frac{C_{\text{target}} - O_{\text{target}}}{O_{\text{target}}}$, $R_{\text{btc}} = \frac{C_{\text{btc}} - O_{\text{btc}}}{O_{\text{btc}}}$
    $\text{btc\_relative\_delta} = R_{\text{target}} - R_{\text{btc}}$

11. **`return_15m`**: Percentage return of current 15m candle.
    $\text{return\_15m} = \frac{C_t - O_t}{O_t}$

## 3. Data Quality Gating
- **NaN, Inf, Null Handling:**
  Any feature vector evaluating to `NaN`, `Inf`, `-Inf`, or `null` triggers automatic rejection of the inference pass for that asset. Telemetry alerting is dispatched.
- **Bounds Validation:**
  - RSI must be within $[0, 100]$.
  - Vol Z-Score must be within $[-10, 100]$.
  - ATR percentage must be $> 0$.
  Failure to meet bounds drops the payload prior to inference.
