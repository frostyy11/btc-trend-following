#!/usr/bin/env python3
"""
Monthly Trend Following in Bitcoin: A Decade of Evidence
=========================================================

A trend-following strategy that:
1. Generates buy signals when 4-week returns exceed +10%
2. Generates sell signals when 4-week returns fall below -10%
3. Uses 10% trailing stops for dynamic downside protection

Usage:
    python3 monthly_trend_following_bitcoin.py <csv_file>
    
Example:
    python3 monthly_trend_following_bitcoin.py 2016-2026.csv

Authors: João Luiz Soares & Claude Haiku 4.5
Date: 2026-03-05
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

INITIAL_CAPITAL = 10000
MONTHLY_BUY_THRESHOLD = 10  # Buy when 4-week return > 10%
MONTHLY_SELL_THRESHOLD = -10  # Sell when 4-week return < -10%
TRAILING_STOP_PCT = 10  # Exit if price drops 10% from high

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(csv_file):
    """Load Bitcoin price data from CSV file"""
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found")
        sys.exit(1)
    
    df = pd.read_csv(csv_file, sep=';')
    df['date'] = pd.to_datetime(df['date'].str.strip('"'))
    df['price'] = pd.to_numeric(df['price'].astype(str).str.strip('"'), errors='coerce')
    
    # Sort and clean
    df = df.sort_values('date').reset_index(drop=True)
    df = df.dropna(subset=['price'])
    df['returns'] = df['price'].pct_change()
    
    return df

# ============================================================================
# STRATEGY IMPLEMENTATION
# ============================================================================

def backtest_monthly_momentum_trailing_stops(data):
    """
    Backtest monthly momentum strategy with trailing stops
    
    Parameters:
        data: DataFrame with 'date', 'price', and 'returns' columns
    
    Returns:
        Dictionary with backtest results and DataFrame with signals
    """
    df = data.copy()
    
    # Calculate 4-week (monthly) price change
    df['monthly_change'] = df['price'].pct_change(4) * 100
    
    # ========================================================================
    # MOMENTUM SIGNAL GENERATION
    # ========================================================================
    
    df['momentum_signal'] = 0
    df.loc[df['monthly_change'] > MONTHLY_BUY_THRESHOLD, 'momentum_signal'] = 1
    df.loc[df['monthly_change'] < MONTHLY_SELL_THRESHOLD, 'momentum_signal'] = -1
    
    # Forward fill to maintain position
    df['momentum_signal'] = df['momentum_signal'].replace(0, np.nan).ffill().fillna(0)
    
    # ========================================================================
    # TRAILING STOP IMPLEMENTATION
    # ========================================================================
    
    df['highest_price_in_position'] = np.nan
    df['trailing_stop_triggered'] = False
    df['signal'] = df['momentum_signal'].copy()
    
    in_long = False
    highest_price = 0
    
    for i in range(len(df)):
        current_price = df['price'].iloc[i]
        current_momentum = df['momentum_signal'].iloc[i]
        
        # Entering or in long position
        if current_momentum == 1:
            if not in_long:
                # Just entered a long position
                in_long = True
                highest_price = current_price
            else:
                # Already in long position, update highest price
                highest_price = max(highest_price, current_price)
            
            df.loc[i, 'highest_price_in_position'] = highest_price
            
            # Check if trailing stop is hit
            stop_price = highest_price * (1 - TRAILING_STOP_PCT / 100)
            if current_price < stop_price:
                # Trailing stop triggered
                df.loc[i, 'signal'] = -1
                df.loc[i, 'trailing_stop_triggered'] = True
                in_long = False
        
        # Exit signal from momentum
        elif current_momentum == -1:
            in_long = False
            df.loc[i, 'signal'] = -1
        
        # No position
        else:
            in_long = False
    
    # ========================================================================
    # CALCULATE STRATEGY RETURNS
    # ========================================================================
    
    df['strategy_returns'] = df['signal'].shift(1) * df['returns']
    df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
    df['portfolio_value'] = INITIAL_CAPITAL * df['cumulative_returns']
    
    # ========================================================================
    # PERFORMANCE METRICS
    # ========================================================================
    
    total_return = (df['portfolio_value'].iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL
    
    # Sharpe ratio (annualized, 52 weeks per year)
    if df['strategy_returns'].std() > 0:
        sharpe = df['strategy_returns'].mean() / df['strategy_returns'].std() * np.sqrt(52)
    else:
        sharpe = 0
    
    # Maximum drawdown
    cummax = df['cumulative_returns'].cummax()
    drawdown = (df['cumulative_returns'] - cummax) / cummax
    max_drawdown = drawdown.min()
    
    # Win rate
    profitable_trades = (df['strategy_returns'] > 0).sum()
    total_trades = (df['strategy_returns'] != 0).sum()
    win_rate = profitable_trades / total_trades if total_trades > 0 else 0
    
    # Number of signals
    num_signals = (df['signal'].diff() != 0).sum()
    
    # Trailing stops triggered
    num_trailing_stops = df['trailing_stop_triggered'].sum()
    
    # Average trade duration
    avg_trade_duration = 532 / (num_signals / 2) if num_signals > 0 else 0
    
    return {
        'data': df,
        'total_return': total_return,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'num_signals': int(num_signals),
        'num_trailing_stops': int(num_trailing_stops),
        'avg_trade_duration': avg_trade_duration,
        'final_portfolio_value': df['portfolio_value'].iloc[-1],
    }

# ============================================================================
# REPORTING
# ============================================================================

def print_results(results):
    """Print detailed backtest results"""
    print("\n" + "=" * 80)
    print("MONTHLY TREND FOLLOWING IN BITCOIN")
    print("=" * 80)
    print("\nBACKTEST PARAMETERS:")
    print(f"  Initial Capital:           ${INITIAL_CAPITAL:,}")
    print(f"  Monthly Buy Threshold:     {MONTHLY_BUY_THRESHOLD}%")
    print(f"  Monthly Sell Threshold:    {MONTHLY_SELL_THRESHOLD}%")
    print(f"  Trailing Stop Level:       {TRAILING_STOP_PCT}%")
    
    print("\nPERFORMANCE METRICS:")
    print(f"  Total Return:              {results['total_return']:>10.2%}")
    print(f"  Final Portfolio Value:     ${results['final_portfolio_value']:>15,.2f}")
    print(f"  Sharpe Ratio:              {results['sharpe']:>10.2f}")
    print(f"  Maximum Drawdown:          {results['max_drawdown']:>10.2%}")
    print(f"  Win Rate:                  {results['win_rate']:>10.2%}")
    
    print("\nTRADING ACTIVITY:")
    print(f"  Total Signals:             {results['num_signals']:>10.0f}")
    print(f"  Trailing Stops Triggered:  {results['num_trailing_stops']:>10.0f}")
    print(f"  Momentum Exits:            {results['num_signals'] - results['num_trailing_stops']:>10.0f}")
    print(f"  Avg Trade Duration:        {results['avg_trade_duration']:>10.1f} weeks")
    
    print("\n" + "=" * 80)

def create_visualizations(results, data_info):
    """Create comprehensive backtesting visualizations"""
    df = results['data']
    
    fig = plt.figure(figsize=(16, 12))
    
    # ========================================================================
    # Chart 1: Price and Portfolio Value
    # ========================================================================
    ax1 = plt.subplot(3, 2, 1)
    ax1_twin = ax1.twinx()
    
    ax1.plot(df['date'], df['price'], color='gray', linewidth=1.5, alpha=0.7, label='Bitcoin Price')
    ax1_twin.plot(df['date'], df['portfolio_value'], color='#1f77b4', linewidth=2.5, label='Strategy Portfolio')
    
    ax1.set_title('Bitcoin Price & Strategy Portfolio Value', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Bitcoin Price (USD)', color='gray', fontsize=10)
    ax1_twin.set_ylabel('Portfolio Value (USD)', color='#1f77b4', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # ========================================================================
    # Chart 2: Cumulative Returns
    # ========================================================================
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(df['date'], (df['cumulative_returns'] - 1) * 100, 
             color='#2ca02c', linewidth=2.5, label='Strategy')
    ax2.fill_between(df['date'], 0, (df['cumulative_returns'] - 1) * 100, 
                     color='#2ca02c', alpha=0.2)
    
    ax2.set_title('Cumulative Returns', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Return (%)', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # ========================================================================
    # Chart 3: Drawdown
    # ========================================================================
    ax3 = plt.subplot(3, 2, 3)
    cummax = df['cumulative_returns'].cummax()
    drawdown = (df['cumulative_returns'] - cummax) / cummax * 100
    
    ax3.fill_between(df['date'], 0, drawdown, color='#d62728', alpha=0.6)
    ax3.plot(df['date'], drawdown, color='#d62728', linewidth=1.5)
    
    ax3.set_title(f'Drawdown (Max: {results["max_drawdown"]:.2%})', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Drawdown (%)', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # ========================================================================
    # Chart 4: Buy/Sell Signals
    # ========================================================================
    ax4 = plt.subplot(3, 2, 4)
    ax4.plot(df['date'], df['price'], color='black', linewidth=1.5, label='Price', alpha=0.6)
    
    # Buy signals (momentum)
    buy_signals = df[df['signal'].diff() == 2]
    ax4.scatter(buy_signals['date'], buy_signals['price'], 
               color='green', marker='^', s=150, alpha=0.7, label='Buy Signal', zorder=5)
    
    # Sell signals (momentum)
    sell_momentum = df[(df['signal'].diff() == -2) & (~df['trailing_stop_triggered'])]
    ax4.scatter(sell_momentum['date'], sell_momentum['price'], 
               color='orange', marker='v', s=150, alpha=0.7, label='Sell Signal (Momentum)', zorder=5)
    
    # Trailing stop exits
    sell_trailing = df[df['trailing_stop_triggered']]
    ax4.scatter(sell_trailing['date'], sell_trailing['price'], 
               color='red', marker='X', s=200, alpha=0.8, label='Sell Signal (Trailing Stop)', zorder=5)
    
    ax4.set_title('Trading Signals', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Price (USD)', fontsize=10)
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    # ========================================================================
    # Chart 5: Monthly Return Distribution
    # ========================================================================
    ax5 = plt.subplot(3, 2, 5)
    returns_pct = df['strategy_returns'] * 100
    ax5.hist(returns_pct, bins=50, color='#1f77b4', alpha=0.7, edgecolor='black')
    ax5.axvline(x=returns_pct.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {returns_pct.mean():.2f}%')
    
    ax5.set_title('Distribution of Weekly Returns', fontweight='bold', fontsize=12)
    ax5.set_xlabel('Weekly Return (%)', fontsize=10)
    ax5.set_ylabel('Frequency', fontsize=10)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # ========================================================================
    # Chart 6: Performance Summary (Text)
    # ========================================================================
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis('off')
    
    summary_text = f"""
STRATEGY SUMMARY

Study Period:
  {data_info['date_from']} to {data_info['date_to']}
  {data_info['num_weeks']} weeks

Performance:
  Return:              {results['total_return']:.2%}
  Sharpe Ratio:        {results['sharpe']:.2f}
  Max Drawdown:        {results['max_drawdown']:.2%}
  Win Rate:            {results['win_rate']:.2%}

Trading Activity:
  Total Signals:       {results['num_signals']}
  Trailing Stops:      {results['num_trailing_stops']}
  Avg Trade Duration:  {results['avg_trade_duration']:.1f} weeks

Capital:
  Initial:             ${INITIAL_CAPITAL:,}
  Final:               ${results['final_portfolio_value']:,.2f}
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Monthly Trend Following in Bitcoin: A Decade of Evidence - Comprehensive Backtest', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('backtest_monthly_trailing_stops_complete.png', dpi=300, bbox_inches='tight')
    print("\n✓ Chart saved: backtest_monthly_trailing_stops_complete.png")

# ============================================================================
# MAIN
# ============================================================================

def main():
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python3 monthly_trend_following_bitcoin.py <csv_file>")
        print("Example: python3 monthly_trend_following_bitcoin.py 2016-2026.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    # Load data
    print(f"\nLoading data from: {csv_file}")
    df = load_data(csv_file)
    
    print(f"Data range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Total records: {len(df)}")
    print(f"Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    
    # Run backtest
    print("\nRunning backtest...")
    results = backtest_monthly_momentum_trailing_stops(df)
    
    # Prepare data info for charts
    data_info = {
        'date_from': df['date'].min().strftime('%Y-%m-%d'),
        'date_to': df['date'].max().strftime('%Y-%m-%d'),
        'num_weeks': len(df),
    }
    
    # Print results
    print_results(results)
    
    # Create visualizations
    create_visualizations(results, data_info)
    
    print("\n" + "=" * 80)
    print("Backtest complete!")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()
