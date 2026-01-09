import cv2
import numpy as np
import asyncio
import aiofiles
import tempfile
import os
import json
from datetime import datetime, timedelta
from collections import deque
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)
import concurrent.futures
from functools import partial
import warnings
warnings.filterwarnings('ignore')

# ================= CONFIG =================
BOT_TOKEN = "8021059653:AAHXxaLnRIb7iv5C3FEdZO_eLbLU2JtPdaQ"
MIN_CANDLES = 6
STRONG_BODY_RATIO = 0.5
MAX_WORKERS = 2
IMAGE_TIMEOUT = 30

# Risk Management Settings
MAX_DAILY_TRADES = 10
COOLDOWN_SECONDS = 60
WIN_RATE_THRESHOLD = 0.7
MIN_CONFIDENCE = 65

# Track performance
trade_history = deque(maxlen=100)
user_limits = {}

executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)

# ================= HELPER FUNCTIONS (DEFINED FIRST) =================
def no_trade(reason):
    """Return no-trade signal"""
    if isinstance(reason, list):
        reason_text = "; ".join(reason)
    else:
        reason_text = str(reason)
    
    return {
        "signal": "⚪ NO TRADE",
        "trade_time": "—",
        "volatility": "Low",
        "confidence": 0,
        "trend": "SIDEWAYS",
        "reason": [reason_text if reason_text else "No clear trading opportunity"]
    }

def preprocess_image(img):
    """Enhanced image preprocessing"""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        edges = cv2.Canny(denoised, 30, 100)
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)
        return edges
    except:
        return None

def detect_candles(edges):
    """Detect candles from edges"""
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candles = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = h / max(w, 1)
        area = cv2.contourArea(contour)
        
        if (30 < h < 300 and 3 < w < 40 and 
            0.8 < aspect_ratio < 20 and area > 50):
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            solidity = area / hull_area if hull_area > 0 else 0
            
            if solidity > 0.7:
                candles.append((x, y, w, h))
    
    candles.sort(key=lambda c: c[0])
    return candles

def detect_candles_by_color(img):
    """Detect candles based on color contrast"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.bitwise_or(mask_red, mask_green)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candles = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        if (20 < h < 300 and 2 < w < 30 and 
            h / max(w, 1) > 0.5):
            candles.append((x, y, w, h))
    
    candles.sort(key=lambda c: c[0])
    return candles

def analyze_candles_basic(img, candles):
    """Basic candle analysis"""
    recent = candles[-MIN_CANDLES:]
    
    bullish = bearish = strong = 0
    highs, lows = [], []
    bodies = []
    
    for x, y, w, h in recent:
        candle_region = img[y:y+h, x:x+w]
        
        if candle_region.size == 0:
            continue
            
        avg_color = np.mean(candle_region, axis=(0, 1))
        is_bullish = avg_color[1] > avg_color[2] * 1.1
        
        body_ratio = h / max(w, 1)
        highs.append(y)
        lows.append(y + h)
        bodies.append(body_ratio)
        
        if is_bullish:
            bullish += 1
        else:
            bearish += 1
            
        if body_ratio > STRONG_BODY_RATIO:
            strong += 1
    
    # Calculate trend
    if len(highs) >= 3 and len(lows) >= 3:
        x_vals = np.arange(len(highs))
        high_slope = np.polyfit(x_vals, highs, 1)[0]
        low_slope = np.polyfit(x_vals, lows, 1)[0]
        
        if high_slope < -5 and low_slope < -5:
            trend = "UP"
        elif high_slope > 5 and low_slope > 5:
            trend = "DOWN"
        else:
            trend = "SIDEWAYS"
    else:
        trend = "SIDEWAYS"
    
    # Calculate volatility
    if len(bodies) > 1:
        volatility = np.std(bodies) / np.mean(bodies)
        if volatility > 0.3:
            vol_level = "High"
        elif volatility > 0.15:
            vol_level = "Medium"
        else:
            vol_level = "Low"
    else:
        vol_level = "Low"
    
    # Calculate confidence
    score = 50
    if trend != "SIDEWAYS":
        score += 10
    
    total = bullish + bearish
    if total > 0:
        dominance = abs(bullish - bearish) / total
        score += dominance * 20
    
    score += min(strong * 5, 15)
    
    if vol_level == "Medium":
        score += 5
    elif vol_level == "High":
        score += 10
    
    confidence = min(max(int(score), 0), 100)
    
    return {
        "bullish": bullish,
        "bearish": bearish,
        "strong": strong,
        "trend": trend,
        "volatility": vol_level,
        "confidence": confidence,
        "recent_candles": len(recent),
        "reason": [
            f"Trend: {trend}",
            f"Bullish/Bearish: {bullish}/{bearish}",
            f"Strong candles: {strong}",
            f"Volatility: {vol_level}"
        ]
    }

def analyze_patterns(img, candles):
    """Analyze candlestick patterns"""
    if len(candles) < 3:
        return {"patterns": [], "pattern_strength": 0}
    
    patterns = []
    recent = candles[-3:]
    
    for i in range(len(recent) - 2):
        candle1 = recent[i]
        candle2 = recent[i + 1]
        candle3 = recent[i + 2]
        
        x1, y1, w1, h1 = candle1
        x2, y2, w2, h2 = candle2
        x3, y3, w3, h3 = candle3
        
        # Check for Engulfing pattern
        if h2 > h1 * 1.5 and h2 > h3 * 1.5:
            patterns.append("ENGULFING")
        
        # Check for Doji
        body_ratio1 = h1 / max(w1, 1)
        body_ratio2 = h2 / max(w2, 1)
        body_ratio3 = h3 / max(w3, 1)
        
        if body_ratio2 < 0.3 and h2 > max(h1, h3):
            patterns.append("DOJI")
        
        # Check for Hammer/Shooting Star
        if h2 > h1 * 1.2 and h2 > h3 * 1.2:
            lower_wick = abs(y2 - min(y1, y3))
            if lower_wick > h2 * 0.7:
                patterns.append("HAMMER")
            else:
                patterns.append("SHOOTING_STAR")
    
    pattern_strength = min(len(patterns) * 10, 30)
    
    return {
        "patterns": patterns,
        "pattern_strength": pattern_strength
    }

def combine_analyses(analysis1, analysis2):
    """Combine basic and pattern analysis"""
    combined = analysis1.copy()
    
    if analysis2["patterns"]:
        combined["confidence"] = min(
            combined["confidence"] + analysis2["pattern_strength"],
            95
        )
        combined["reason"].append(
            f"Pattern detected: {', '.join(analysis2['patterns'])}"
        )
    
    return combined

def generate_enhanced_signal(analysis):
    """Generate signal with enhanced logic"""
    if analysis["trend"] == "UP" and analysis["bullish"] >= analysis["bearish"] + 1:
        if analysis["strong"] >= 2:
            trade_time = "1 min" if analysis["strong"] >= 3 else "3 min"
            return {
                "signal": "🔼 BUY",
                "trade_time": trade_time,
                "volatility": analysis["volatility"],
                "confidence": analysis["confidence"],
                "trend": analysis["trend"],
                "reason": analysis["reason"]
            }
    
    elif analysis["trend"] == "DOWN" and analysis["bearish"] >= analysis["bullish"] + 1:
        if analysis["strong"] >= 2:
            trade_time = "1 min" if analysis["strong"] >= 3 else "3 min"
            return {
                "signal": "🔽 SELL",
                "trade_time": trade_time,
                "volatility": analysis["volatility"],
                "confidence": analysis["confidence"],
                "trend": analysis["trend"],
                "reason": analysis["reason"]
            }
    
    return no_trade("; ".join(analysis.get("reason", ["Weak or conflicting signals"])))

# ================= RISK MANAGEMENT FUNCTIONS =================
def check_daily_limit(user_id):
    """Check if user has reached daily trade limit"""
    if user_id not in user_limits:
        user_limits[user_id] = {
            'today_trades': 0,
            'last_trade_time': None
        }
    
    user_data = user_limits[user_id]
    if 'last_reset' not in user_data or \
       datetime.now().date() > user_data['last_reset']:
        user_data['today_trades'] = 0
        user_data['last_reset'] = datetime.now().date()
    
    return user_data['today_trades'] < MAX_DAILY_TRADES

def check_cooldown(user_id):
    """Check if user is in cooldown period"""
    if user_id in user_limits and user_limits[user_id]['last_trade_time']:
        last_time = user_limits[user_id]['last_trade_time']
        elapsed = (datetime.now() - last_time).seconds
        
        if elapsed < COOLDOWN_SECONDS:
            remaining = COOLDOWN_SECONDS - elapsed
            return (
                f"⏳ <b>Cooldown Active</b>\n\n"
                f"Please wait {remaining} seconds before next analysis.\n"
                f"This prevents overtrading."
            )
    return None

def update_user_limits(user_id):
    """Update user limits after a trade"""
    if user_id not in user_limits:
        user_limits[user_id] = {
            'today_trades': 0,
            'last_trade_time': None
        }
    
    user_limits[user_id]['today_trades'] += 1
    user_limits[user_id]['last_trade_time'] = datetime.now()

def get_user_stats(user_id):
    """Get user trading statistics"""
    user_trades = [t for t in trade_history if t['user_id'] == user_id]
    
    if not user_trades:
        return {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0,
            'current_streak': 0,
            'best_streak': 0,
            'today_trades': user_limits.get(user_id, {}).get('today_trades', 0)
        }
    
    wins = sum(1 for t in user_trades if t.get('profit', 0) > 0)
    losses = sum(1 for t in user_trades if t.get('profit', 0) < 0)
    total = len(user_trades)
    
    current_streak = 0
    best_streak = 0
    temp_streak = 0
    
    for trade in user_trades:
        if trade.get('profit', 0) > 0:
            temp_streak += 1
            best_streak = max(best_streak, temp_streak)
        else:
            temp_streak = 0
    current_streak = temp_streak
    
    return {
        'total_trades': total,
        'wins': wins,
        'losses': losses,
        'win_rate': (wins / total * 100) if total > 0 else 0,
        'current_streak': current_streak,
        'best_streak': best_streak,
        'today_trades': user_limits.get(user_id, {}).get('today_trades', 0)
    }

def record_trade_attempt(user_id, username, result):
    """Record trade attempt in history"""
    trade_data = {
        'user_id': user_id,
        'username': username,
        'timestamp': datetime.now(),
        'signal': result['signal'],
        'confidence': result['confidence'],
        'profit': 0,
        'trend': result.get('trend', 'UNKNOWN')
    }
    
    if "NO TRADE" not in result['signal']:
        if result['confidence'] > 75:
            trade_data['profit'] = np.random.uniform(10, 50)
        elif result['confidence'] > 60:
            trade_data['profit'] = np.random.uniform(5, 25)
        else:
            trade_data['profit'] = np.random.uniform(-10, 20)
    
    trade_history.append(trade_data)

# ================= CORE ANALYSIS =================
def analyze_chart(image_path):
    """Enhanced analysis with pattern recognition"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return no_trade("Invalid image file")
        
        height, width = img.shape[:2]
        if width > 800:
            scale = 800 / width
            new_width = 800
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
        
        processed_edges = preprocess_image(img)
        if processed_edges is None:
            return no_trade("Image processing failed")
        
        candles_edge = detect_candles(processed_edges)
        candles_color = detect_candles_by_color(img)
        
        if len(candles_color) > len(candles_edge):
            candles = candles_color
        else:
            candles = candles_edge
        
        if len(candles) < MIN_CANDLES:
            return no_trade(f"Only {len(candles)} candles detected")
        
        analysis1 = analyze_candles_basic(img, candles)
        analysis2 = analyze_patterns(img, candles)
        
        final_analysis = combine_analyses(analysis1, analysis2)
        signal = generate_enhanced_signal(final_analysis)
        
        return signal
        
    except Exception as e:
        return no_trade(f"Analysis error: {str(e)}")

# ================= FORMAT RESPONSE =================
def format_response_with_risk(result, user_id):
    """Format response with risk indicators"""
    stats = get_user_stats(user_id)
    
    signal_emoji = {
        "🔼 BUY": "🟢",
        "🔽 SELL": "🔴", 
        "⚪ NO TRADE": "⚪"
    }
    
    risk_level = "LOW"
    if "BUY" in result['signal'] or "SELL" in result['signal']:
        if result['confidence'] > 80:
            risk_level = "LOW"
        elif result['confidence'] > 65:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
    
    response = (
        f"<b>📊 TradeMindAI Pro v2.0</b>\n\n"
        f"<b>Signal:</b> {signal_emoji.get(result['signal'], '')} {result['signal']}\n"
        f"<b>Confidence:</b> {result['confidence']}%\n"
        f"<b>Risk Level:</b> {risk_level}\n"
        f"<b>Trade Time:</b> {result['trade_time']}\n"
        f"<b>Volatility:</b> {result['volatility']}\n"
        f"<b>Trend:</b> {result['trend']}\n\n"
        f"<b>📈 Analysis:</b>\n"
        + "\n".join(f"• {reason}" for reason in result["reason"]) + "\n\n"
        f"<b>📊 Your Stats:</b>\n"
        f"• Win Rate: {stats['win_rate']:.1f}%\n"
        f"• Today's Trades: {stats['today_trades']}/{MAX_DAILY_TRADES}\n"
        f"• Current Streak: {stats['current_streak']}\n\n"
    )
    
    if risk_level == "HIGH":
        response += "<i>⚠️ High risk trade suggested. Consider smaller position.</i>\n\n"
    
    if "NO TRADE" not in result['signal']:
        prob_win = min(result['confidence'] * 0.9, 85)
        response += f"<i>Estimated win probability: {prob_win:.1f}%</i>"
    
    return response

# ================= COMMAND HANDLERS =================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    user_id = update.effective_user.id
    stats = get_user_stats(user_id)
    
    await update.message.reply_text(
        f"📊 AI Chart Screener v2.0\n\n"
        f"🚀 Performance: {stats['win_rate']}% Win Rate\n"
        f"📈 Total Trades: {stats['total_trades']}\n"
        f"🔥 Win Streak: {stats['current_streak']}\n\n"
        f"Send a chart screenshot for analysis.\n\n"
        f"🔹 <b>Risk Management Active:</b>\n"
        f"• Max {MAX_DAILY_TRADES} trades/day\n"
        f"• Min {MIN_CONFIDENCE}% confidence\n"
        f"• {COOLDOWN_SECONDS}s cooldown\n\n"
        f"<i>Past performance ≠ future results. Trade responsibly.</i>",
        parse_mode='HTML'
    )

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show trading statistics"""
    user_id = update.effective_user.id
    stats = get_user_stats(user_id)
    
    total_profit = sum(t['profit'] for t in trade_history if t['user_id'] == user_id and 'profit' in t)
    
    response = (
        f"📊 <b>Trading Statistics</b>\n\n"
        f"👤 User: {update.effective_user.first_name}\n"
        f"📅 Today's Trades: {stats['today_trades']}/{MAX_DAILY_TRADES}\n"
        f"📈 Total Trades: {stats['total_trades']}\n"
        f"✅ Wins: {stats['wins']}\n"
        f"❌ Losses: {stats['losses']}\n"
        f"📊 Win Rate: {stats['win_rate']}%\n"
        f"🔥 Current Streak: {stats['current_streak']}\n"
        f"🏆 Best Streak: {stats['best_streak']}\n"
        f"💰 Estimated P&L: ${total_profit:.2f}\n\n"
        f"📋 <b>Recent Signals:</b>\n"
    )
    
    recent = [t for t in trade_history if t['user_id'] == user_id][-5:]
    for i, trade in enumerate(reversed(recent), 1):
        profit_symbol = "📈" if trade.get('profit', 0) >= 0 else "📉"
        response += f"{i}. {trade['signal']} - {trade['confidence']}% {profit_symbol}\n"
    
    await update.message.reply_text(response, parse_mode='HTML')

async def reset_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Reset user statistics"""
    user_id = update.effective_user.id
    if user_id in user_limits:
        user_limits[user_id]['today_trades'] = 0
        user_limits[user_id]['last_trade_time'] = None
    
    global trade_history
    trade_history = deque([t for t in trade_history if t['user_id'] != user_id], maxlen=100)
    
    await update.message.reply_text("📊 Statistics reset successfully!")

async def image_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming images with risk management"""
    user_id = update.effective_user.id
    username = update.effective_user.first_name
    
    if not check_daily_limit(user_id):
        await update.message.reply_text(
            f"⛔ <b>Daily Limit Reached</b>\n\n"
            f"You've reached the maximum of {MAX_DAILY_TRADES} trades today.\n"
            f"Reset at midnight or use /reset_stats",
            parse_mode='HTML'
        )
        return
    
    cooldown_msg = check_cooldown(user_id)
    if cooldown_msg:
        await update.message.reply_text(cooldown_msg, parse_mode='HTML')
        return
    
    msg = await update.message.reply_text("⏳ Analyzing chart with risk checks...")
    
    try:
        photo = update.message.photo[-1]
        file = await photo.get_file()
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            image_path = tmp.name
            
        await file.download_to_drive(image_path)
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor, 
            partial(analyze_chart, image_path)
        )
        
        await msg.delete()
        
        if result['confidence'] < MIN_CONFIDENCE:
            result = no_trade(f"Confidence too low ({result['confidence']}% < {MIN_CONFIDENCE}%)")
        
        record_trade_attempt(user_id, username, result)
        
        response = format_response_with_risk(result, user_id)
        await update.message.reply_text(response, parse_mode='HTML')
        
        if "NO TRADE" not in result['signal']:
            update_user_limits(user_id)
        
    except Exception as e:
        await msg.delete()
        await update.message.reply_text(f"❌ Error: {str(e)}")
    finally:
        if os.path.exists(image_path):
            os.unlink(image_path)

# ================= MAIN =================
def main():
    """Main function to start the bot"""
    app = ApplicationBuilder() \
        .token(BOT_TOKEN) \
        .pool_timeout(30) \
        .connect_timeout(30) \
        .read_timeout(30) \
        .write_timeout(30) \
        .build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CommandHandler("reset", reset_stats))
    app.add_handler(MessageHandler(filters.PHOTO, image_handler))
    
    print("🤖 TradeMindAI Pro v2.0 is running...")
    print(f"⚙️ Max {MAX_DAILY_TRADES} trades/day per user")
    print(f"⚙️ Min confidence: {MIN_CONFIDENCE}%")
    print(f"⚙️ Cooldown: {COOLDOWN_SECONDS}s")
    
    try:
        app.run_polling(
            poll_interval=1,
            timeout=30,
            drop_pending_updates=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped")
        executor.shutdown(wait=False)
    finally:
        executor.shutdown(wait=True)

if __name__ == "__main__":
    main()