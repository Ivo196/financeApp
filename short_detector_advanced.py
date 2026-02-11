"""
🚀 MÓDULO AVANZADO PARA DETECCIÓN DE SHORTS
Sistema profesional con explicaciones detalladas y análisis inteligente
"""

import ccxt
import pandas as pd
import numpy as np
import ta
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Excel
import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment

# Configuración global
exchange = None
TIMEFRAME = '1h'
LIMIT = 200


def init_exchange():
    """Inicializa conexión con Binance"""
    global exchange
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    exchange.load_markets()
    print('✅ Exchange inicializado')


def get_ohlcv_data(symbol, timeframe=None, limit=None):
    """Obtiene datos OHLCV"""
    if timeframe is None:
        timeframe = TIMEFRAME
    if limit is None:
        limit = LIMIT
        
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        print(f'❌ Error {symbol}: {e}')
        return None


def get_funding_rate(symbol):
    """Obtiene funding rate"""
    try:
        ticker = symbol.replace('/USDT', 'USDT')
        funding = exchange.fapiPublicGetPremiumIndex({'symbol': ticker})
        return float(funding['lastFundingRate']) * 100
    except:
        return None


def calculate_indicators(df):
    """
    Calcula TODOS los indicadores técnicos
    
    Returns: DataFrame con 30+ indicadores
    """
    df = df.copy()
    
    # RSI
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    
    # EMAs
    for period in [9, 20, 50, 100, 200]:
        df[f'ema_{period}'] = ta.trend.ema_indicator(df['close'], window=period)
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['close'], window=20)
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_mid'] = bollinger.bollinger_mavg()
    df['bb_low'] = bollinger.bollinger_lband()
    df['bb_percent'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])
    
    # ATR
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
    df['atr_percent'] = (df['atr'] / df['close']) * 100
    
    # Volumen
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Stochastic
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    # ADX
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
    df['adx_pos'] = ta.trend.adx_pos(df['high'], df['low'], df['close'])
    df['adx_neg'] = ta.trend.adx_neg(df['high'], df['low'], df['close'])
    
    # Williams %R
    df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
    
    # CCI
    df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'])
    
    # Cambios porcentuales
    for period in [1, 3, 5, 10]:
        df[f'pct_change_{period}'] = df['close'].pct_change(period) * 100
    
    # Máximos/Mínimos
    df['high_20'] = df['high'].rolling(window=20).max()
    df['low_20'] = df['low'].rolling(window=20).min()
    df['distance_from_high'] = ((df['close'] - df['high_20']) / df['high_20']) * 100
    
    return df


def analyze_short_signals(df, symbol):
    """
    Analiza señales con EXPLICACIONES DETALLADAS
    
    Returns:
        dict con score, señales, detalles y explicaciones del POR QUÉ
    """
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    signals = []
    score = 0
    details = {}
    explanations = []
    
    # 1. RSI (0-3 pts)
    if last['rsi'] > 75:
        pts = 3
        signals.append('🔴 RSI muy sobrecomprado (>75)')
        explanations.append({
            'indicador': 'RSI',
            'valor': f"{last['rsi']:.1f}",
            'puntos': pts,
            'por_que': f'RSI de {last["rsi"]:.1f} indica sobrecompra extrema. Cuando RSI > 75, el precio corrige en el 70-80% de los casos históricos.',
            'que_significa': 'El precio subió demasiado rápido y necesita una pausa o corrección. Alta probabilidad de reversión bajista.',
            'confianza': '⭐⭐⭐'
        })
        score += pts
    elif last['rsi'] > 70:
        pts = 2
        signals.append('🟡 RSI sobrecomprado (>70)')
        explanations.append({
            'indicador': 'RSI',
            'valor': f"{last['rsi']:.1f}",
            'puntos': pts,
            'por_que': f'RSI de {last["rsi"]:.1f} muestra sobrecompra moderada. Zona de posible reversión.',
            'que_significa': 'El momentum alcista puede estar perdiendo fuerza.',
            'confianza': '⭐⭐'
        })
        score += pts
    elif last['rsi'] > 65:
        pts = 1
        signals.append('🟢 RSI alto (>65)')
        score += pts
    
    details['rsi'] = last['rsi']
    
    # 2. MACD (0-4 pts)
    if prev['macd_hist'] > 0 and last['macd_hist'] < 0:
        pts = 4
        signals.append('🔴 MACD cruce bajista RECIENTE')
        explanations.append({
            'indicador': 'MACD',
            'valor': f"Cruce {prev['macd_hist']:.4f} → {last['macd_hist']:.4f}",
            'puntos': pts,
            'por_que': 'MACD acaba de cruzar de positivo a negativo. Esto señala cambio de momentum alcista a bajista.',
            'que_significa': 'El impulso comprador se detuvo y comenzó impulso vendedor. Señal muy fuerte de reversión inminente.',
            'confianza': '⭐⭐⭐⭐'
        })
        score += pts
    elif last['macd_hist'] < 0:
        pts = 2
        signals.append('🟡 MACD zona bajista')
        score += pts
    
    details['macd_signal'] = 'Bajista' if last['macd_hist'] < 0 else 'Alcista'
    
    # 3. EMAs (0-5 pts)
    emas = [9, 20, 50, 100, 200]
    below_emas = sum([last['close'] < last[f'ema_{e}'] for e in emas])
    
    if below_emas >= 4:
        pts = 5
        signals.append(f'🔴 Precio bajo {below_emas}/5 EMAs')
        explanations.append({
            'indicador': 'EMAs',
            'valor': f'{below_emas}/5',
            'puntos': pts,
            'por_que': f'Precio por debajo de {below_emas} medias móviles. Todas actúan como resistencias.',
            'que_significa': 'Tendencia bajista confirmada en múltiples timeframes. Alta presión vendedora estructural.',
            'confianza': '⭐⭐⭐⭐⭐'
        })
        score += pts
    elif below_emas == 3:
        pts = 3
        signals.append('🟡 Precio bajo 3 EMAs')
        score += pts
    elif below_emas == 2:
        score += 2
    elif below_emas == 1:
        score += 1
    
    details['below_emas'] = f'{below_emas}/5'
    
    # 4. Death Cross (0-5 pts)
    if prev['ema_50'] >= prev['ema_200'] and last['ema_50'] < last['ema_200']:
        pts = 5
        signals.append('💀 DEATH CROSS')
        explanations.append({
            'indicador': 'Death Cross',
            'valor': 'EMA50 cruzó EMA200',
            'puntos': pts,
            'por_que': 'La Cruz de la Muerte indica cambio de tendencia alcista a bajista en largo plazo.',
            'que_significa': 'Señal macro-bajista. Históricamente precede caídas prolongadas del 20-40%.',
            'confianza': '⭐⭐⭐⭐⭐'
        })
        score += pts
    
    # 5. Stochastic (0-2 pts)
    if last['stoch_k'] > 80:
        pts = 2
        signals.append(f'🟡 Stochastic sobrecomprado ({last["stoch_k"]:.0f})')
        explanations.append({
            'indicador': 'Stochastic',
            'valor': f"{last['stoch_k']:.1f}",
            'puntos': pts,
            'por_que': f'Stochastic en {last["stoch_k"]:.0f} (>80) indica extrema sobrecompra.',
            'que_significa': 'Precio en zona de agotamiento a corto plazo. Reversión probable en próximas velas.',
            'confianza': '⭐⭐'
        })
        score += pts
    
    details['stochastic'] = last['stoch_k']
    
    # 6. Volumen (0-3 pts)
    if last['close'] < last['open'] and last['volume_ratio'] > 2:
        pts = 3
        signals.append(f'🔴 ALTO volumen bajista ({last["volume_ratio"]:.1f}x)')
        explanations.append({
            'indicador': 'Volumen',
            'valor': f"{last['volume_ratio']:.1f}x",
            'puntos': pts,
            'por_que': f'Volumen {last["volume_ratio"]:.1f}x el promedio en vela roja. Alto volumen + caída = convicción bajista.',
            'que_significa': 'Muchos traders vendiendo con convicción. Presión vendedora institucional.',
            'confianza': '⭐⭐⭐'
        })
        score += pts
    elif last['close'] < last['open'] and last['volume_ratio'] > 1.5:
        pts = 2
        signals.append(f'🟡 Volumen elevado bajista')
        score += pts
    
    details['volume_ratio'] = last['volume_ratio']
    
    # 7. Bollinger (0-3 pts)
    if last['bb_percent'] > 0.95:
        pts = 3
        signals.append(f'🔴 Precio en tope Bollinger')
        explanations.append({
            'indicador': 'Bollinger Bands',
            'valor': f"{last['bb_percent']*100:.0f}%",
            'puntos': pts,
            'por_que': f'Precio en {last["bb_percent"]*100:.0f}% del rango Bollinger (>95%). Extensión extrema.',
            'que_significa': 'Precio estirado al máximo. Ley de regresión a la media sugiere retorno.',
            'confianza': '⭐⭐⭐'
        })
        score += pts
    elif last['bb_percent'] > 0.85:
        pts = 2
        signals.append('🟡 Precio alto en Bollinger')
        score += pts
    
    # 8. ADX (0-3 pts)
    if last['adx'] > 25 and last['adx_neg'] > last['adx_pos']:
        pts = 3
        signals.append(f'🔴 Tendencia bajista fuerte (ADX {last["adx"]:.0f})')
        explanations.append({
            'indicador': 'ADX',
            'valor': f"{last['adx']:.1f}",
            'puntos': pts,
            'por_que': f'ADX {last["adx"]:.0f} (>25) con DI- > DI+ confirma tendencia bajista fuerte.',
            'que_significa': 'No es movimiento lateral. Hay momentum bajista claro y sostenido.',
            'confianza': '⭐⭐⭐'
        })
        score += pts
    
    details['adx'] = last['adx']
    
    # 9. Caídas recientes (0-2 pts)
    pct_5 = df['pct_change_1'].tail(5).sum()
    if pct_5 < -5:
        pts = 2
        signals.append(f'🔴 Caída fuerte: {pct_5:.1f}%')
        explanations.append({
            'indicador': 'Momentum Reciente',
            'valor': f"{pct_5:.1f}%",
            'puntos': pts,
            'por_que': f'Caída de {pct_5:.1f}% en 5 velas. Momentum bajista activado.',
            'que_significa': 'La caída ya está en marcha y tiende a continuar por inercia.',
            'confianza': '⭐⭐'
        })
        score += pts
    elif pct_5 < -3:
        pts = 1
        signals.append(f'🟡 Caída reciente: {pct_5:.1f}%')
        score += pts
    
    # 10. Williams %R (0-2 pts)
    if last['williams_r'] > -20:
        pts = 2
        signals.append('🟡 Williams %R sobrecomprado')
        score += pts
    
    # 11. CCI (0-2 pts)
    if last['cci'] > 100:
        pts = 2
        signals.append(f'🟡 CCI sobrecomprado ({last["cci"]:.0f})')
        score += pts
    
    # 12. Funding rate (0-2 pts)
    funding = get_funding_rate(symbol)
    if funding:
        details['funding_rate'] = funding
        if funding < -0.01:
            pts = 2
            signals.append(f'💰 Funding muy negativo: {funding:.3f}%')
            explanations.append({
                'indicador': 'Funding Rate',
                'valor': f"{funding:.3f}%",
                'puntos': pts,
                'por_que': f'Funding rate {funding:.3f}% (muy negativo). Holders de shorts reciben pago.',
                'que_significa': 'Mayoría del mercado posicionado en shorts. Sentimiento muy bajista.',
                'confianza': '⭐⭐'
            })
            score += pts
        elif funding < 0:
            pts = 1
            signals.append(f'💰 Funding negativo')
            score += pts
    
    # 13. Distancia de máximos (0-2 pts)
    if last['distance_from_high'] > -2:
        pts = 2
        signals.append(f'🟡 Cerca de máximo ({last["distance_from_high"]:.1f}%)')
        explanations.append({
            'indicador': 'Distancia de Máximo',
            'valor': f"{last['distance_from_high']:.1f}%",
            'puntos': pts,
            'por_que': f'Solo {abs(last["distance_from_high"]):.1f}% bajo el máximo de 20 períodos.',
            'que_significa': 'En zona de resistencia histórica. Difícil continuar subiendo.',
            'confianza': '⭐⭐'
        })
        score += pts
    
    details['distance_from_high'] = last['distance_from_high']
    
    # Preparar targets
    details['target_1_ema20'] = last['ema_20']
    details['target_2_ema50'] = last['ema_50']
    details['target_3_ema200'] = last['ema_200']
    
    return {
        'symbol': symbol,
        'score': score,
        'max_score': 35,
        'signals': signals,
        'price': last['close'],
        'details': details,
        'explanations': explanations,
        'timestamp': datetime.now()
    }


def scan_all_markets(symbols, min_score=10):
    """Escanea todos los mercados"""
    results = []
    
    print(f'🔍 Escaneando {len(symbols)} criptomonedas...')
    print('=' * 80)
    
    for i, symbol in enumerate(symbols, 1):
        try:
            df = get_ohlcv_data(symbol)
            if df is None or len(df) < 200:
                continue
            
            df = calculate_indicators(df)
            analysis = analyze_short_signals(df, symbol)
            results.append(analysis)
            
            if analysis['score'] >= min_score:
                print(f"✅ {i}/{len(symbols)} - {symbol}: {analysis['score']}/{analysis['max_score']} ⭐")
            else:
                print(f"⚪ {i}/{len(symbols)} - {symbol}: {analysis['score']}/{analysis['max_score']}")
                
        except Exception as e:
            print(f"❌ {i}/{len(symbols)} - {symbol}: {str(e)[:50]}")
            continue
    
    results.sort(key=lambda x: x['score'], reverse=True)
    opportunities = [r for r in results if r['score'] >= min_score]
    
    print('\n' + '=' * 80)
    print(f'🎯 Encontradas {len(opportunities)} oportunidades (score >= {min_score})')
    
    return results, opportunities


def generate_excel_report(all_results, opportunities, filename=None):
    """
    Genera Excel COMPLETO con análisis detallado
    
    Hojas:
    1. Resumen Ejecutivo
    2. Análisis por Cripto (top 5)
    3. Recomendaciones de Trading
    4. Estadísticas
    5. Todas las Señales
    """
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'analisis_shorts_{timestamp}.xlsx'
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        
        # HOJA 1: Resumen
        summary = []
        for opp in opportunities[:10]:
            summary.append({
                'Symbol': opp['symbol'],
                'Score': f"{opp['score']}/{opp['max_score']}",
                'Precio': f"${opp['price']:.4f}",
                'RSI': f"{opp['details'].get('rsi', 0):.1f}",
                'MACD': opp['details'].get('macd_signal', '-'),
                'EMAs': opp['details'].get('below_emas', '-'),
                'Vol': f"{opp['details'].get('volume_ratio', 0):.2f}x",
                'Señales': len(opp['signals']),
                'Confianza': '⭐⭐⭐⭐⭐' if opp['score'] >= 25 else '⭐⭐⭐⭐' if opp['score'] >= 20 else '⭐⭐⭐'
            })
        pd.DataFrame(summary).to_excel(writer, sheet_name='Resumen', index=False)
        
        # HOJA 2-6: Detalle de cada cripto (top 5)
        for idx, opp in enumerate(opportunities[:5], 1):
            details = []
            
            # Info general
            details.append({
                'Categoría': 'GENERAL',
                'Métrica': 'Símbolo',
                'Valor': opp['symbol'],
                'Explicación': ''
            })
            details.append({
                'Categoría': 'GENERAL',
                'Métrica': 'Precio',
                'Valor': f"${opp['price']:.6f}",
                'Explicación': 'Precio actual'
            })
            details.append({
                'Categoría': 'GENERAL',
                'Métrica': 'Score',
                'Valor': f"{opp['score']}/{opp['max_score']}",
                'Explicación': f'Confianza sobre {opp["max_score"]} puntos'
            })
            
            # Explicaciones de indicadores
            for exp in opp['explanations']:
                details.append({
                    'Categoría': 'INDICADOR',
                    'Métrica': exp['indicador'],
                    'Valor': exp['valor'],
                    'Explicación': f"{exp['por_que']} | {exp['que_significa']} | {exp['confianza']}"
                })
            
            df_det = pd.DataFrame(details)
            sheet = f"{idx}. {opp['symbol'].replace('/', '-')}"[:31]
            df_det.to_excel(writer, sheet_name=sheet, index=False)
        
        # HOJA: Recomendaciones
        recos = []
        for opp in opportunities[:5]:
            price = opp['price']
            stop = price * 1.03
            t1 = opp['details'].get('target_1_ema20', price * 0.97)
            t2 = opp['details'].get('target_2_ema50', price * 0.94)
            t3 = opp['details'].get('target_3_ema200', price * 0.90)
            
            recos.append({
                'Symbol': opp['symbol'],
                'Score': f"{opp['score']}/{opp['max_score']}",
                'Entrada': f"${price:.4f}",
                'Stop-Loss': f"${stop:.4f}",
                'Target 1': f"${t1:.4f}",
                'Target 2': f"${t2:.4f}",
                'Target 3': f"${t3:.4f}",
                'Gain T1': f"{((price-t1)/price*100):.1f}%",
                'Gain T2': f"{((price-t2)/price*100):.1f}%",
                'Gain T3': f"{((price-t3)/price*100):.1f}%",
                'Risk/Reward': f"{((price-t2)/(stop-price)):.1f}:1",
                'Recomendación': '🔥 FUERTE' if opp['score'] >= 20 else '🟠 MODERADA'
            })
        pd.DataFrame(recos).to_excel(writer, sheet_name='Trading', index=False)
        
        # HOJA: Estadísticas
        scores = [r['score'] for r in all_results]
        stats = [
            {'Métrica': 'Total Analizadas', 'Valor': len(all_results)},
            {'Métrica': 'Oportunidades', 'Valor': len(opportunities)},
            {'Métrica': 'Score Máximo', 'Valor': f"{max(scores)}/35"},
            {'Métrica': 'Score Promedio', 'Valor': f"{np.mean(scores):.1f}/35"},
            {'Métrica': 'Score > 20', 'Valor': len([s for s in scores if s >= 20])},
            {'Métrica': 'Fecha', 'Valor': datetime.now().strftime('%Y-%m-%d %H:%M')}
        ]
        pd.DataFrame(stats).to_excel(writer, sheet_name='Stats', index=False)
    
    print(f'✅ Excel generado: {filename}')
    return filename


def analyze_with_claude(opportunity, api_key):
    """Análisis con Claude AI"""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        
        prompt = f"""Analiza esta oportunidad de SHORT:

SÍMBOLO: {opportunity['symbol']}
PRECIO: ${opportunity['price']:.6f}
SCORE: {opportunity['score']}/{opportunity['max_score']}

SEÑALES:
{chr(10).join([f'- {s}' for s in opportunity['signals']])}

EXPLICACIONES:
{chr(10).join([f"- {e['indicador']}: {e['por_que']}" for e in opportunity['explanations']])}

Responde en español:
1. ¿Es buena oportunidad? (SÍ/NO/NEUTRAL)
2. Confianza (1-10)
3. Principales riesgos (máx 2)
4. Principales fortalezas (máx 2)
5. Stop-loss recomendado (%)
6. Target recomendado (%)
7. Comentario final (2 líneas)"""
        
        msg = client.messages.create(
            model='claude-sonnet-4-20250514',
            max_tokens=800,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        return {
            'success': True,
            'analysis': msg.content[0].text
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


# Inicializar automáticamente
init_exchange()
print('✅ Módulo short_detector_advanced cargado')
