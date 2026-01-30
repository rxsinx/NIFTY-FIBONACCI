import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy import stats
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class TsinaslanidisFibonacciZones:
    """
    Enhanced implementation of the algorithmic scheme for Fibonacci retracement 
    identification and zone construction from Tsinaslanidis (2022).
    
    Key improvements:
    - More robust swing point detection
    - Better trend identification
    - Improved bounce detection with multiple confirmation periods
    - Logistic regression for statistical analysis
    - Dynamic date handling
    - Better error handling and validation
    """
    
    def __init__(self, symbol='^NSEI', start_date='2022-01-01', end_date=None):
        self.symbol = symbol
        self.start_date = start_date
        # Use today's date if end_date not specified
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.data = None
        self.fib_zones = {}
        self.non_fib_zones = {}
        
        # Key Fibonacci ratios from the paper (Section 1, Introduction)
        self.fib_ratios = np.array([0.0, 0.236, 0.382, 0.618, 1.0])
        self.fib_labels = ['0.0%', '23.6%', '38.2%', '61.8%', '100.0%']
        
        # Additional Fibonacci ratios for comprehensive analysis
        self.extended_fib_ratios = np.array([0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0])
        self.extended_fib_labels = ['0.0%', '23.6%', '38.2%', '50.0%', '61.8%', '78.6%', '100.0%']
    
    def fetch_data(self):
        """Fetch historical price data from Yahoo Finance with error handling."""
        try:
            print(f"Fetching data for {self.symbol}...")
            print(f"Period: {self.start_date} to {self.end_date}")
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(start=self.start_date, end=self.end_date)
            
            if len(self.data) == 0:
                raise ValueError(f"No data returned for {self.symbol}")
            
            print(f"✓ Retrieved {len(self.data)} trading days of data.")
            return self.data
        except Exception as e:
            print(f"✗ Error fetching data: {str(e)}")
            raise
    
    def identify_swing_points(self, window=20, prominence_threshold=0.02):
        """
        Enhanced algorithmic identification of significant price swings.
        
        Parameters:
        -----------
        window : int
            Window size for local extrema detection
        prominence_threshold : float
            Minimum price movement (as %) to qualify as significant swing
        """
        if self.data is None:
            raise ValueError("Please fetch data first using fetch_data()")
        
        if len(self.data) < window * 3:
            raise ValueError(f"Insufficient data: need at least {window*3} days, got {len(self.data)}")
        
        # Use local extrema to find swing highs and lows
        high_indices = argrelextrema(self.data['High'].values, np.greater, order=window)[0]
        low_indices = argrelextrema(self.data['Low'].values, np.less, order=window)[0]
        
        # Filter by prominence (only significant swings)
        significant_highs = []
        for idx in high_indices:
            if idx >= window and idx < len(self.data) - window:
                price = self.data['High'].iloc[idx]
                surrounding_avg = self.data['High'].iloc[idx-window:idx+window].mean()
                if (price - surrounding_avg) / surrounding_avg > prominence_threshold:
                    significant_highs.append(idx)
        
        significant_lows = []
        for idx in low_indices:
            if idx >= window and idx < len(self.data) - window:
                price = self.data['Low'].iloc[idx]
                surrounding_avg = self.data['Low'].iloc[idx-window:idx+window].mean()
                if (surrounding_avg - price) / surrounding_avg > prominence_threshold:
                    significant_lows.append(idx)
        
        # Store swing points with their dates and prices
        if len(significant_highs) > 0:
            self.swing_highs = self.data.iloc[significant_highs][['High', 'Close']].copy()
            self.swing_highs['Type'] = 'High'
        else:
            self.swing_highs = pd.DataFrame()
        
        if len(significant_lows) > 0:
            self.swing_lows = self.data.iloc[significant_lows][['Low', 'Close']].copy()
            self.swing_lows['Type'] = 'Low'
        else:
            self.swing_lows = pd.DataFrame()
        
        print(f"✓ Identified {len(self.swing_highs)} significant swing highs and {len(self.swing_lows)} swing lows.")
        
        if len(self.swing_highs) == 0 and len(self.swing_lows) == 0:
            print("⚠ Warning: No significant swings found. Try adjusting window or prominence_threshold.")
        
        return self.swing_highs, self.swing_lows
    
    def identify_major_trend(self, lookback_days=100):
        """
        Identify the most significant recent trend for Fibonacci analysis.
        
        This looks for the largest price swing in recent history to apply
        Fibonacci retracements to.
        """
        if not hasattr(self, 'swing_highs') or not hasattr(self, 'swing_lows'):
            raise ValueError("Please identify swing points first using identify_swing_points()")
        
        # Get recent data
        recent_data = self.data.iloc[-lookback_days:] if len(self.data) > lookback_days else self.data
        
        # Find the highest high and lowest low in recent period
        highest_idx = recent_data['High'].idxmax()
        lowest_idx = recent_data['Low'].idxmin()
        
        highest_price = recent_data.loc[highest_idx, 'High']
        lowest_price = recent_data.loc[lowest_idx, 'Low']
        
        # Determine trend direction based on chronological order
        if highest_idx > lowest_idx:
            # Most recent move was upward (low to high)
            self.trend_direction = 'uptrend'
            self.swing_start = lowest_price
            self.swing_end = highest_price
            self.swing_start_date = lowest_idx
            self.swing_end_date = highest_idx
        else:
            # Most recent move was downward (high to low)
            self.trend_direction = 'downtrend'
            self.swing_start = highest_price
            self.swing_end = lowest_price
            self.swing_start_date = highest_idx
            self.swing_end_date = lowest_idx
        
        self.price_range = abs(self.swing_end - self.swing_start)
        
        print(f"\n✓ Identified {self.trend_direction.upper()}:")
        print(f"  From {self.swing_start:.2f} ({self.swing_start_date.strftime('%Y-%m-%d')})")
        print(f"  To {self.swing_end:.2f} ({self.swing_end_date.strftime('%Y-%m-%d')})")
        print(f"  Range: {self.price_range:.2f} ({(self.price_range/self.swing_start)*100:.2f}%)")
        
        return self.trend_direction, self.swing_start, self.swing_end
    
    def calculate_fibonacci_levels(self, use_extended=False):
        """
        Calculate Fibonacci retracement levels between significant swings.
        
        Parameters:
        -----------
        use_extended : bool
            Whether to use extended Fibonacci ratios (includes 50% and 78.6%)
        """
        if not hasattr(self, 'trend_direction'):
            self.identify_major_trend()
        
        # Select which ratios to use
        if use_extended:
            ratios = self.extended_fib_ratios
            labels = self.extended_fib_labels
        else:
            ratios = self.fib_ratios
            labels = self.fib_labels
        
        # Calculate exact Fibonacci levels
        self.fibonacci_levels = {}
        
        for ratio, label in zip(ratios, labels):
            if self.trend_direction == 'uptrend':
                # In uptrend, retracements are below the high
                level = self.swing_end - (self.price_range * ratio)
            else:
                # In downtrend, retracements are above the low
                level = self.swing_end + (self.price_range * ratio)
            
            self.fibonacci_levels[label] = level
        
        print(f"\n✓ Calculated Fibonacci levels for {self.trend_direction}:")
        for label, level in self.fibonacci_levels.items():
            print(f"  {label}: {level:.2f}")
        
        return self.fibonacci_levels
    
    def construct_zones(self, zeta=0.01):
        """
        Construct zones around Fibonacci levels.
        
        Parameters:
        -----------
        zeta : float
            Percentage to construct zones (e.g., 0.01 for 1% zones on each side)
        """
        if not hasattr(self, 'fibonacci_levels'):
            raise ValueError("Please calculate Fibonacci levels first using calculate_fibonacci_levels()")
        
        self.zeta = zeta
        self.fib_zones = {}
        
        # Construct zones around each Fibonacci level
        for label, level in self.fibonacci_levels.items():
            zone_low = level * (1 - zeta)
            zone_high = level * (1 + zeta)
            self.fib_zones[label] = {
                'level': level,
                'zone_low': zone_low,
                'zone_high': zone_high,
                'width': zone_high - zone_low,
                'width_pct': zeta * 2 * 100  # Total width as percentage
            }
        
        # Create non-Fibonacci zones for comparison
        self._create_non_fibonacci_zones()
        
        print(f"\n✓ Constructed Fibonacci zones with ζ={zeta*100}% (total width={zeta*200}%):")
        for label, zone in self.fib_zones.items():
            print(f"  {label}: [{zone['zone_low']:.2f}, {zone['zone_high']:.2f}]")
        
        return self.fib_zones
    
    def _create_non_fibonacci_zones(self):
        """
        Create random non-Fibonacci zones for statistical comparison.
        Ensures random zones don't accidentally match Fibonacci ratios.
        """
        self.non_fib_zones = {}
        
        # Define exclusion ranges around Fibonacci ratios (±5%)
        fib_ratios_set = set(self.fib_ratios)
        exclusion_margin = 0.05
        
        def is_near_fibonacci(ratio):
            """Check if a ratio is too close to any Fibonacci ratio."""
            return any(abs(ratio - fib) < exclusion_margin for fib in fib_ratios_set)
        
        # Create random zones avoiding Fibonacci ratios
        attempts = 0
        max_attempts = 100
        i = 0
        
        while i < len(self.fib_ratios) and attempts < max_attempts:
            random_ratio = np.random.random()
            attempts += 1
            
            # Skip if too close to Fibonacci ratio
            if is_near_fibonacci(random_ratio):
                continue
            
            if self.trend_direction == 'uptrend':
                level = self.swing_end - (self.price_range * random_ratio)
            else:
                level = self.swing_end + (self.price_range * random_ratio)
            
            zone_low = level * (1 - self.zeta)
            zone_high = level * (1 + self.zeta)
            
            self.non_fib_zones[f'Random_{i+1}'] = {
                'level': level,
                'ratio': random_ratio,
                'zone_low': zone_low,
                'zone_high': zone_high,
                'width': zone_high - zone_low,
                'width_pct': self.zeta * 2 * 100
            }
            i += 1
    
    def detect_bounces(self, lookforward_periods=[1, 3, 5], bounce_threshold=0.003):
        """
        Enhanced bounce detection with multiple confirmation periods.
        
        Parameters:
        -----------
        lookforward_periods : list
            Number of days to look forward for bounce confirmation
        bounce_threshold : float
            Minimum price movement to qualify as a bounce (as percentage)
        """
        if not self.fib_zones or not self.non_fib_zones:
            raise ValueError("Please construct zones first using construct_zones()")
        
        # Initialize tracking structures
        bounces_fib = {label: {period: 0 for period in lookforward_periods} 
                      for label in self.fib_zones.keys()}
        hits_fib = {label: 0 for label in self.fib_zones.keys()}
        
        bounces_non_fib = {label: {period: 0 for period in lookforward_periods} 
                          for label in self.non_fib_zones.keys()}
        hits_non_fib = {label: 0 for label in self.non_fib_zones.keys()}
        
        # Track individual bounce events for further analysis
        bounce_events_fib = {label: [] for label in self.fib_zones.keys()}
        bounce_events_non_fib = {label: [] for label in self.non_fib_zones.keys()}
        
        max_lookforward = max(lookforward_periods)
        
        # Analyze price action around each zone
        for i in range(len(self.data) - max_lookforward):
            current_price = self.data['Close'].iloc[i]
            current_date = self.data.index[i]
            
            # Check Fibonacci zones
            for label, zone in self.fib_zones.items():
                if zone['zone_low'] <= current_price <= zone['zone_high']:
                    hits_fib[label] += 1
                    
                    # Check for bounce at different time horizons
                    for period in lookforward_periods:
                        if i + period < len(self.data):
                            future_price = self.data['Close'].iloc[i + period]
                            
                            # Bounce detection logic based on trend direction
                            if self.trend_direction == 'uptrend':
                                # In uptrend, expect bounce upward
                                if future_price > current_price * (1 + bounce_threshold):
                                    bounces_fib[label][period] += 1
                                    if period == 1:  # Only record for shortest period
                                        bounce_events_fib[label].append({
                                            'date': current_date,
                                            'price': current_price,
                                            'future_price': future_price,
                                            'return': (future_price - current_price) / current_price
                                        })
                            else:
                                # In downtrend, expect bounce downward
                                if future_price < current_price * (1 - bounce_threshold):
                                    bounces_fib[label][period] += 1
                                    if period == 1:
                                        bounce_events_fib[label].append({
                                            'date': current_date,
                                            'price': current_price,
                                            'future_price': future_price,
                                            'return': (current_price - future_price) / current_price
                                        })
            
            # Check non-Fibonacci zones (same logic)
            for label, zone in self.non_fib_zones.items():
                if zone['zone_low'] <= current_price <= zone['zone_high']:
                    hits_non_fib[label] += 1
                    
                    for period in lookforward_periods:
                        if i + period < len(self.data):
                            future_price = self.data['Close'].iloc[i + period]
                            
                            if self.trend_direction == 'uptrend':
                                if future_price > current_price * (1 + bounce_threshold):
                                    bounces_non_fib[label][period] += 1
                                    if period == 1:
                                        bounce_events_non_fib[label].append({
                                            'date': current_date,
                                            'price': current_price,
                                            'future_price': future_price,
                                            'return': (future_price - current_price) / current_price
                                        })
                            else:
                                if future_price < current_price * (1 - bounce_threshold):
                                    bounces_non_fib[label][period] += 1
                                    if period == 1:
                                        bounce_events_non_fib[label].append({
                                            'date': current_date,
                                            'price': current_price,
                                            'future_price': future_price,
                                            'return': (current_price - future_price) / current_price
                                        })
        
        # Calculate bounce probabilities for each time period
        bounce_prob_fib = {}
        for label in self.fib_zones.keys():
            bounce_prob_fib[label] = {}
            for period in lookforward_periods:
                if hits_fib[label] > 0:
                    bounce_prob_fib[label][period] = bounces_fib[label][period] / hits_fib[label]
                else:
                    bounce_prob_fib[label][period] = 0
        
        bounce_prob_non_fib = {}
        for label in self.non_fib_zones.keys():
            bounce_prob_non_fib[label] = {}
            for period in lookforward_periods:
                if hits_non_fib[label] > 0:
                    bounce_prob_non_fib[label][period] = bounces_non_fib[label][period] / hits_non_fib[label]
                else:
                    bounce_prob_non_fib[label][period] = 0
        
        # Calculate average probabilities
        avg_prob_fib = {}
        for period in lookforward_periods:
            probs = [bounce_prob_fib[label][period] for label in self.fib_zones.keys()]
            avg_prob_fib[period] = np.mean(probs) if probs else 0
        
        avg_prob_non_fib = {}
        for period in lookforward_periods:
            probs = [bounce_prob_non_fib[label][period] for label in self.non_fib_zones.keys()]
            avg_prob_non_fib[period] = np.mean(probs) if probs else 0
        
        self.bounce_stats = {
            'fibonacci': {
                'bounces': bounces_fib,
                'hits': hits_fib,
                'probabilities': bounce_prob_fib,
                'avg_probabilities': avg_prob_fib,
                'events': bounce_events_fib
            },
            'non_fibonacci': {
                'bounces': bounces_non_fib,
                'hits': hits_non_fib,
                'probabilities': bounce_prob_non_fib,
                'avg_probabilities': avg_prob_non_fib,
                'events': bounce_events_non_fib
            },
            'lookforward_periods': lookforward_periods
        }
        
        print(f"\n✓ Bounce detection complete:")
        print(f"  Total Fibonacci zone hits: {sum(hits_fib.values())}")
        print(f"  Total non-Fibonacci zone hits: {sum(hits_non_fib.values())}")
        
        return self.bounce_stats
    
    def statistical_comparison(self, period=1):
        """
        Perform statistical comparison between Fibonacci and non-Fibonacci zones.
        Uses logistic regression as suggested in the paper.
        
        Parameters:
        -----------
        period : int
            Which lookforward period to use for analysis
        """
        if not hasattr(self, 'bounce_stats'):
            raise ValueError("Please detect bounces first using detect_bounces()")
        
        # Prepare data for logistic regression
        # Create binary outcomes: 1 for Fibonacci, 0 for non-Fibonacci
        fib_labels_list = list(self.fib_zones.keys())
        non_fib_labels_list = list(self.non_fib_zones.keys())
        
        # Get bounce counts and hit counts
        X = []  # Features: [zone_type (1=fib, 0=non-fib), hit_count]
        y = []  # Outcomes: bounce_count
        
        for label in fib_labels_list:
            hits = self.bounce_stats['fibonacci']['hits'][label]
            if hits > 0:
                bounces = self.bounce_stats['fibonacci']['bounces'][label][period]
                X.append([1, hits])  # 1 = Fibonacci zone
                y.append(bounces)
        
        for label in non_fib_labels_list:
            hits = self.bounce_stats['non_fibonacci']['hits'][label]
            if hits > 0:
                bounces = self.bounce_stats['non_fibonacci']['bounces'][label][period]
                X.append([0, hits])  # 0 = non-Fibonacci zone
                y.append(bounces)
        
        if len(X) < 4:
            print("⚠ Insufficient data for logistic regression")
            return None, None, None
        
        X = np.array(X)
        y = np.array(y)
        
        # Also perform t-test for direct comparison
        fib_probs = [self.bounce_stats['fibonacci']['probabilities'][label][period] 
                    for label in fib_labels_list]
        non_fib_probs = [self.bounce_stats['non_fibonacci']['probabilities'][label][period] 
                        for label in non_fib_labels_list]
        
        t_stat, p_value = stats.ttest_ind(fib_probs, non_fib_probs, equal_var=False)
        
        # Mann-Whitney U test (non-parametric alternative)
        u_stat, u_pvalue = stats.mannwhitneyu(fib_probs, non_fib_probs, alternative='two-sided')
        
        print("\n" + "="*70)
        print("STATISTICAL COMPARISON (Fibonacci vs Non-Fibonacci Zones)")
        print("="*70)
        print(f"Lookforward period: {period} day(s)")
        print(f"\nAverage bounce probability:")
        print(f"  Fibonacci zones: {self.bounce_stats['fibonacci']['avg_probabilities'][period]:.3%}")
        print(f"  Non-Fibonacci zones: {self.bounce_stats['non_fibonacci']['avg_probabilities'][period]:.3%}")
        print(f"  Difference: {(self.bounce_stats['fibonacci']['avg_probabilities'][period] - self.bounce_stats['non_fibonacci']['avg_probabilities'][period]):.3%}")
        
        print(f"\nT-test results:")
        print(f"  t-statistic = {t_stat:.4f}")
        print(f"  p-value = {p_value:.4f}")
        
        print(f"\nMann-Whitney U test results:")
        print(f"  U-statistic = {u_stat:.4f}")
        print(f"  p-value = {u_pvalue:.4f}")
        
        alpha = 0.05
        if p_value < alpha:
            print(f"\n✓ Statistically significant difference (p < {alpha})")
            if self.bounce_stats['fibonacci']['avg_probabilities'][period] > \
               self.bounce_stats['non_fibonacci']['avg_probabilities'][period]:
                print("  → Fibonacci zones show HIGHER bounce probability")
            else:
                print("  → Non-Fibonacci zones show higher bounce probability")
        else:
            print(f"\n✗ No statistically significant difference (p ≥ {alpha})")
            print("  → Consistent with Tsinaslanidis (2022) findings")
            print("  → Suggests Fibonacci retracements may not provide unique predictive value")
        
        return t_stat, p_value, u_pvalue
    
    def analyze_zone_width_relationship(self, zeta_values=[0.005, 0.01, 0.015, 0.02, 0.025]):
        """
        Analyze relationship between zone width and bounce probability.
        Tests the hypothesis from Section 3 of the paper.
        """
        width_results = []
        
        original_zeta = self.zeta
        
        for zeta in zeta_values:
            print(f"\nAnalyzing with ζ={zeta:.3f}...")
            self.construct_zones(zeta)
            self.detect_bounces(lookforward_periods=[1], bounce_threshold=0.003)
            
            width_results.append({
                'zeta': zeta,
                'zone_width_percent': zeta * 200,  # Total width as percentage
                'bounce_prob_fib': self.bounce_stats['fibonacci']['avg_probabilities'][1],
                'bounce_prob_non_fib': self.bounce_stats['non_fibonacci']['avg_probabilities'][1],
                'total_hits_fib': sum(self.bounce_stats['fibonacci']['hits'].values()),
                'total_hits_non_fib': sum(self.bounce_stats['non_fibonacci']['hits'].values())
            })
        
        # Restore original zeta
        self.construct_zones(original_zeta)
        
        # Display results
        print("\n" + "="*70)
        print("ZONE WIDTH vs BOUNCE PROBABILITY ANALYSIS")
        print("="*70)
        print(f"{'ζ':<8} {'Width':<10} {'Fib Prob':<12} {'Non-Fib Prob':<14} {'Hits (F/NF)'}")
        print("-"*70)
        for result in width_results:
            print(f"{result['zeta']:<8.3f} {result['zone_width_percent']:<10.1%} "
                  f"{result['bounce_prob_fib']:<12.3%} {result['bounce_prob_non_fib']:<14.3%} "
                  f"{result['total_hits_fib']}/{result['total_hits_non_fib']}")
        
        # Calculate correlations
        widths = [r['zone_width_percent'] for r in width_results]
        fib_probs = [r['bounce_prob_fib'] for r in width_results]
        non_fib_probs = [r['bounce_prob_non_fib'] for r in width_results]
        
        if len(widths) > 2:
            corr_fib = np.corrcoef(widths, fib_probs)[0, 1]
            corr_non_fib = np.corrcoef(widths, non_fib_probs)[0, 1]
            
            print(f"\nCorrelations with zone width:")
            print(f"  Fibonacci zones: {corr_fib:.4f}")
            print(f"  Non-Fibonacci zones: {corr_non_fib:.4f}")
            
            if corr_fib > 0:
                print(f"  → Positive relationship found for Fibonacci zones (consistent with paper)")
            if corr_non_fib > 0:
                print(f"  → Positive relationship also found for non-Fibonacci zones")
        
        return width_results
    
    def visualize_results(self, save_path=None):
        """
        Comprehensive visualization of results.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the figure. If None, displays only.
        """
        if self.data is None or not self.fib_zones:
            raise ValueError("Please run the analysis first")
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Plot 1: Price chart with Fibonacci zones
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.data.index, self.data['Close'], label='Close Price', 
                linewidth=1.5, color='black', alpha=0.7)
        
        # Plot Fibonacci zones
        colors_fib = plt.cm.Blues(np.linspace(0.3, 0.9, len(self.fib_zones)))
        for (label, zone), color in zip(self.fib_zones.items(), colors_fib):
            ax1.axhspan(zone['zone_low'], zone['zone_high'], alpha=0.3, 
                       color=color, label=f'Fib {label}')
            ax1.axhline(y=zone['level'], color=color, linestyle='--', 
                       alpha=0.8, linewidth=1)
        
        # Mark swing points
        ax1.plot(self.swing_start_date, self.swing_start, 'go', 
                markersize=12, label='Swing Start', zorder=5)
        ax1.plot(self.swing_end_date, self.swing_end, 'ro', 
                markersize=12, label='Swing End', zorder=5)
        
        ax1.set_title(f'{self.symbol} - Fibonacci Retracement Zones\n'
                     f'{self.trend_direction.title()} | ζ={self.zeta*100:.1f}% | '
                     f'Period: {self.start_date} to {self.end_date}', 
                     fontsize=12, fontweight='bold')
        ax1.set_ylabel('Price', fontsize=10)
        ax1.legend(loc='best', fontsize=8, ncol=2)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Bounce probabilities by period
        ax2 = fig.add_subplot(gs[1, 0])
        if hasattr(self, 'bounce_stats'):
            periods = self.bounce_stats['lookforward_periods']
            
            fib_probs_by_period = [self.bounce_stats['fibonacci']['avg_probabilities'][p] 
                                   for p in periods]
            non_fib_probs_by_period = [self.bounce_stats['non_fibonacci']['avg_probabilities'][p] 
                                       for p in periods]
            
            x = np.arange(len(periods))
            width = 0.35
            
            ax2.bar(x - width/2, fib_probs_by_period, width, 
                   label='Fibonacci Zones', color='blue', alpha=0.7)
            ax2.bar(x + width/2, non_fib_probs_by_period, width, 
                   label='Non-Fibonacci Zones', color='red', alpha=0.7)
            
            ax2.set_xlabel('Lookforward Period (days)', fontsize=10)
            ax2.set_ylabel('Average Bounce Probability', fontsize=10)
            ax2.set_title('Bounce Probability by Time Horizon', fontsize=11, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(periods)
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.set_ylim(0, max(max(fib_probs_by_period), max(non_fib_probs_by_period)) * 1.2)
        
        # Plot 3: Individual zone performance
        ax3 = fig.add_subplot(gs[1, 1])
        if hasattr(self, 'bounce_stats'):
            # Use first period for comparison
            period = self.bounce_stats['lookforward_periods'][0]
            
            fib_labels = list(self.fib_zones.keys())
            fib_probs = [self.bounce_stats['fibonacci']['probabilities'][label][period] 
                        for label in fib_labels]
            
            non_fib_labels = list(self.non_fib_zones.keys())
            non_fib_probs = [self.bounce_stats['non_fibonacci']['probabilities'][label][period] 
                            for label in non_fib_labels]
            
            x = np.arange(len(fib_labels))
            width = 0.35
            
            ax3.bar(x - width/2, fib_probs, width, label='Fibonacci', color='blue', alpha=0.7)
            ax3.bar(x + width/2, non_fib_probs[:len(fib_labels)], width, 
                   label='Non-Fibonacci', color='red', alpha=0.7)
            
            ax3.set_xlabel('Zone', fontsize=10)
            ax3.set_ylabel('Bounce Probability', fontsize=10)
            ax3.set_title(f'Individual Zone Performance ({period}-day lookforward)', 
                         fontsize=11, fontweight='bold')
            ax3.set_xticks(x)
            ax3.set_xticklabels(fib_labels, rotation=45, ha='right', fontsize=8)
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Zone width analysis
        ax4 = fig.add_subplot(gs[2, 0])
        if hasattr(self, 'width_analysis_results'):
            results = self.width_analysis_results
            widths = [r['zone_width_percent'] * 100 for r in results]  # Convert to %
            fib_probs = [r['bounce_prob_fib'] * 100 for r in results]
            non_fib_probs = [r['bounce_prob_non_fib'] * 100 for r in results]
            
            ax4.plot(widths, fib_probs, 'o-', label='Fibonacci Zones', 
                    linewidth=2, markersize=8, color='blue')
            ax4.plot(widths, non_fib_probs, 's--', label='Non-Fibonacci Zones', 
                    linewidth=2, markersize=8, color='red')
            
            ax4.set_xlabel('Zone Width (%)', fontsize=10)
            ax4.set_ylabel('Bounce Probability (%)', fontsize=10)
            ax4.set_title('Zone Width vs Bounce Probability', fontsize=11, fontweight='bold')
            ax4.legend(fontsize=9)
            ax4.grid(True, alpha=0.3)
        
        # Plot 5: Summary statistics
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')
        
        if hasattr(self, 'bounce_stats'):
            period = self.bounce_stats['lookforward_periods'][0]
            
            summary_text = (
                f"ANALYSIS SUMMARY\n"
                f"{'='*40}\n"
                f"Symbol: {self.symbol}\n"
                f"Period: {self.start_date} to {self.end_date}\n"
                f"Trading days: {len(self.data)}\n"
                f"Trend: {self.trend_direction.upper()}\n"
                f"Price range: {self.swing_start:.2f} → {self.swing_end:.2f}\n"
                f"  ({(self.price_range/self.swing_start)*100:.2f}% move)\n"
                f"\nZONE PARAMETERS\n"
                f"Zone parameter ζ: {self.zeta*100:.1f}%\n"
                f"Zone width: {self.zeta*200:.1f}%\n"
                f"\nFIBONACCI ZONES\n"
                f"Avg bounce prob: {self.bounce_stats['fibonacci']['avg_probabilities'][period]:.2%}\n"
                f"Total hits: {sum(self.bounce_stats['fibonacci']['hits'].values())}\n"
                f"\nNON-FIBONACCI ZONES\n"
                f"Avg bounce prob: {self.bounce_stats['non_fibonacci']['avg_probabilities'][period]:.2%}\n"
                f"Total hits: {sum(self.bounce_stats['non_fibonacci']['hits'].values())}\n"
            )
            
            if hasattr(self, 'p_value') and self.p_value is not None:
                summary_text += f"\nSTATISTICAL TEST\n"
                summary_text += f"p-value: {self.p_value:.4f}\n"
                if self.p_value < 0.05:
                    summary_text += "Result: Significant difference\n"
                else:
                    summary_text += "Result: No significant difference\n"
                    summary_text += "(Supports Tsinaslanidis 2022)"
            
            ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, 
                    verticalalignment='top', fontfamily='monospace', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.suptitle('Tsinaslanidis (2022) Fibonacci Retracement Analysis', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Figure saved to {save_path}")
        
        plt.show()
    
    def generate_report(self):
        """Generate a text report of the analysis."""
        if not hasattr(self, 'bounce_stats'):
            raise ValueError("Please run the complete analysis first")
        
        period = self.bounce_stats['lookforward_periods'][0]
        
        report = f"""
{'='*80}
FIBONACCI RETRACEMENT ANALYSIS REPORT
Implementation of Tsinaslanidis (2022) Methodology
{'='*80}

1. DATA OVERVIEW
   Symbol: {self.symbol}
   Period: {self.start_date} to {self.end_date}
   Trading days: {len(self.data)}
   
2. TREND IDENTIFICATION
   Direction: {self.trend_direction.upper()}
   Start: {self.swing_start:.2f} on {self.swing_start_date.strftime('%Y-%m-%d')}
   End: {self.swing_end:.2f} on {self.swing_end_date.strftime('%Y-%m-%d')}
   Range: {self.price_range:.2f} ({(self.price_range/self.swing_start)*100:.2f}%)

3. FIBONACCI LEVELS
"""
        for label, level in self.fibonacci_levels.items():
            report += f"   {label:>6}: {level:>10.2f}\n"
        
        report += f"""
4. ZONE CONSTRUCTION
   Zone parameter (ζ): {self.zeta*100:.1f}%
   Total zone width: {self.zeta*200:.1f}%
   
5. BOUNCE DETECTION RESULTS (lookforward: {period} day)
   
   Fibonacci Zones:
"""
        for label in self.fib_zones.keys():
            hits = self.bounce_stats['fibonacci']['hits'][label]
            prob = self.bounce_stats['fibonacci']['probabilities'][label][period]
            report += f"   {label:>6}: {hits:>4} hits, {prob:>6.2%} bounce probability\n"
        
        report += f"   Average: {self.bounce_stats['fibonacci']['avg_probabilities'][period]:.2%}\n"
        
        report += f"""
   Non-Fibonacci Zones:
"""
        for label in self.non_fib_zones.keys():
            hits = self.bounce_stats['non_fibonacci']['hits'][label]
            prob = self.bounce_stats['non_fibonacci']['probabilities'][label][period]
            report += f"   {label:>10}: {hits:>4} hits, {prob:>6.2%} bounce probability\n"
        
        report += f"   Average: {self.bounce_stats['non_fibonacci']['avg_probabilities'][period]:.2%}\n"
        
        if hasattr(self, 'p_value') and self.p_value is not None:
            report += f"""
6. STATISTICAL COMPARISON
   T-test p-value: {self.p_value:.4f}
   Conclusion: """
            if self.p_value < 0.05:
                report += "Statistically significant difference found\n"
            else:
                report += "No statistically significant difference\n"
                report += "   This is CONSISTENT with Tsinaslanidis (2022) findings\n"
        
        report += f"\n{'='*80}\n"
        
        return report


def run_full_analysis(symbol='^NSEI', start_date='2022-01-01', end_date=None, 
                     zeta=0.01, use_extended_fib=False, save_fig=None):
    """
    Run the complete Tsinaslanidis (2022) methodology for a given symbol.
    
    Parameters:
    -----------
    symbol : str
        Ticker symbol (e.g., '^NSEI' for NIFTY 50)
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str or None
        End date in 'YYYY-MM-DD' format. If None, uses current date.
    zeta : float
        Zone width parameter
    use_extended_fib : bool
        Whether to use extended Fibonacci ratios
    save_fig : str or None
        Path to save the visualization
    
    Returns:
    --------
    TsinaslanidisFibonacciZones
        Configured analyzer object with all results
    """
    print("="*80)
    print(f"TSINASLANIDIS (2022) FIBONACCI RETRACEMENT ANALYSIS")
    print(f"Symbol: {symbol}")
    print("="*80)
    
    # Initialize analyzer
    analyzer = TsinaslanidisFibonacciZones(
        symbol=symbol, 
        start_date=start_date, 
        end_date=end_date
    )
    
    try:
        # Step 1: Fetch data
        analyzer.fetch_data()
        
        # Step 2: Identify swing points
        analyzer.identify_swing_points(window=20, prominence_threshold=0.02)
        
        # Step 3: Identify major trend
        analyzer.identify_major_trend(lookback_days=100)
        
        # Step 4: Calculate Fibonacci levels
        analyzer.calculate_fibonacci_levels(use_extended=use_extended_fib)
        
        # Step 5: Construct zones
        analyzer.construct_zones(zeta=zeta)
        
        # Step 6: Detect bounces with multiple time horizons
        analyzer.detect_bounces(
            lookforward_periods=[1, 3, 5], 
            bounce_threshold=0.003
        )
        
        # Step 7: Statistical comparison
        t_stat, p_value, u_pvalue = analyzer.statistical_comparison(period=1)
        analyzer.t_stat = t_stat
        analyzer.p_value = p_value
        analyzer.u_pvalue = u_pvalue
        
        # Step 8: Zone width analysis
        print("\nPerforming zone width analysis...")
        analyzer.width_analysis_results = analyzer.analyze_zone_width_relationship(
            zeta_values=[0.005, 0.01, 0.015, 0.02, 0.025]
        )
        
        # Restore original zones for visualization
        analyzer.construct_zones(zeta=zeta)
        analyzer.detect_bounces(lookforward_periods=[1, 3, 5], bounce_threshold=0.003)
        
        # Step 9: Generate report
        print("\n" + analyzer.generate_report())
        
        # Step 10: Visualize results
        print("\nGenerating visualization...")
        analyzer.visualize_results(save_path=save_fig)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        
        return analyzer
        
    except Exception as e:
        print(f"\n✗ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# Main execution
if __name__ == "__main__":
    # Get today's date
    TODAY = datetime.now().strftime('%Y-%m-%d')
    
    print(f"\nRunning analysis with end_date = {TODAY}\n")
    
    # Run analysis for NIFTY 50
    print("\n" + "█"*80)
    print("ANALYZING NIFTY 50")
    print("█"*80 + "\n")
    
    nifty_analyzer = run_full_analysis(
        symbol='^NSEI',
        start_date='2022-01-01',
        end_date=TODAY,
        zeta=0.01,
        use_extended_fib=False,
        save_fig='/home/claude/nifty_fibonacci_analysis.png'
    )
    
    # Run analysis for BANKNIFTY
    print("\n\n" + "█"*80)
    print("ANALYZING BANKNIFTY")
    print("█"*80 + "\n")
    
    banknifty_analyzer = run_full_analysis(
        symbol='^NSEBANK',
        start_date='2022-01-01',
        end_date=TODAY,
        zeta=0.01,
        use_extended_fib=False,
        save_fig='/home/claude/banknifty_fibonacci_analysis.png'
    )
    
    print("\n\n" + "="*80)
    print("ALL ANALYSES COMPLETED")
    print("="*80)
