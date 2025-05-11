# Trading-On-Trends
An Advanced Stock Sentiment Analysis system
#### 1. Using APIs in the project (Criterion 11)
   * **Reddit API**: Social media sentiment data collection using PRAW library
     - Implementation: `StockSentimentAnalyzer.fetch_reddit_data()`
     - Features: Subreddit search, post collection, sentiment scoring
   * **Yahoo Finance API**: Historical stock market data retrieval using yfinance
     - Implementation: `StockSentimentAnalyzer.fetch_stock_data()`
     - Features: OHLCV data, technical indicators, price history
   * **Flask API**: Web application endpoints for analysis requests
     - Implementation: `/analyze` route in Flask application
     - Real-time analysis and visualization generation

#### 2. Data cleaning and/or Data transformation (Criterion 3)
   * **Text preprocessing**:
     - URL removal, special character handling, whitespace normalization
     - Implementation: `StockSentimentAnalyzer.clean_text()`
   * **Data merging**:
     - Stock and sentiment data fusion with date alignment
     - Implementation: `StockSentimentAnalyzer.merge_stock_and_sentiment()`
   * **Missing value handling**:
     - NaN and infinity value replacement with appropriate defaults
     - Forward/backward fill for price data, zero-fill for sentiment data
   * **Feature engineering**:
     - Technical indicators (RSI, MACD, Bollinger Bands)
     - Lag features, rolling statistics, interaction features
     - Implementation: `StockSentimentAnalyzer.add_technical_indicators()`

#### 3. Logistic Regression and variants (Criterion 9)
   * **Binary classification** for stock price direction prediction (up/down)
   * **Model variants implemented**:
     - Standard Logistic Regression with L2 regularization
     - Random Forest Classifier
     - XGBoost Classifier
   * **Implementation**: `StockSentimentAnalyzer.train_models()`
   * **Evaluation**: `StockSentimentAnalyzer._evaluate_logistic_model()`
     - Metrics: Accuracy, Precision, Recall, F1-score, AUC-ROC
     - Confusion matrix visualization

### Additional Criteria Implemented

#### 1. Object-oriented code (Criterion 1)
   * **Main class**: `StockSentimentAnalyzer`
   * **Methods organized by functionality**:
     - Data collection: `fetch_stock_data()`, `fetch_reddit_data()`
     - Data processing: `clean_text()`, `merge_stock_and_sentiment()`
     - Analysis: `analyze_data()`, `train_models()`
     - Visualization: Private methods for plotting
   * **Encapsulation**: Private helper methods, state management

#### 2. Regular Expression (Criterion 5)
   * **Text cleaning patterns**:
     - URL removal: `r'https?://\S+|www\.\S+'`
     - Username extraction: `r'@\w+'`
     - Hashtag normalization: `r'#'`
     - Special character filtering: `r'[^\w\s\.,!?]'`
   * **Implementation**: `StockSentimentAnalyzer.clean_text()`

#### 3. Linear Regression and variants (Criterion 8)
   * **Continuous target prediction** for stock returns percentage
   * **Model variants implemented**:
     - Ridge Regression (L2 regularization)
     - Gradient Boosting Regressor
     - XGBoost Regressor
   * **Implementation**: `StockSentimentAnalyzer.train_models()`
   * **Evaluation**: `StockSentimentAnalyzer._evaluate_linear_model()`
     - Metrics: RMSE, MAE, R², Directional Accuracy

### New Features and Enhancements

#### 1. Interactive Web Application
   * **Flask-based API**: Real-time analysis endpoint
   * **Dynamic visualizations**: Plotly charts with interactive elements
   * **Multi-panel dashboard**: Price charts, volume, technical indicators

#### 2. Advanced Sentiment Analysis
   * **Financial lexicon enhancement**: Custom financial terms for VADER
   * **Multi-source aggregation**: Combined sentiment from multiple subreddits
   * **Temporal features**: Sentiment momentum, rolling averages

#### 3. Comprehensive Technical Analysis
   * **Indicators implemented**:
     - Simple/Exponential Moving Averages (SMA/EMA)
     - Relative Strength Index (RSI)
     - MACD (Moving Average Convergence Divergence)
     - Bollinger Bands
     - Average True Range (ATR)
     - Stochastic Oscillator
   * **Visualization**: Multi-panel technical charts

#### 4. Enhanced Model Training Pipeline
   * **Feature selection**: Random Forest-based importance ranking
   * **Cross-validation**: Time series split for temporal data
   * **Class balancing**: SMOTE for handling imbalanced datasets
   * **Model selection**: Automated comparison of multiple algorithms

#### 5. Robust Error Handling and Data Validation
   * **NaN/infinity handling**: Comprehensive cleaning before analysis
   * **API error management**: Graceful fallbacks for data collection
   * **Visualization safety**: Validated inputs for plotting functions

#### 6. Performance Monitoring
   * **Model metrics tracking**: JSON storage of evaluation results
   * **Feature importance analysis**: Visual comparison across models
   * **Backtesting framework**: Historical performance validation

#### 7. Scalable Architecture
   * **Modular design**: Separate concerns for data, models, visualization
   * **Configurable parameters**: Easy adjustment of analysis parameters
   * **Extensible framework**: Simple addition of new data sources or models

### Project Structure
stock_sentiment_analysis/
├── data/
│   ├── raw/              # Original API data
│   ├── processed/        # Cleaned and merged data
│   └── final/            # Analysis-ready datasets
├── models/               # Trained model artifacts
├── visualizations/       # Generated charts and plots
├── app.py               # Flask web application
├── StockSentimentAnalyzer.py  # Main analysis class
└── requirements.txt      # Project dependencies


### Future Enhancements
1. Real-time streaming data integration
2. Advanced deep learning models (LSTM, Transformer)
3. Multi-asset portfolio analysis
4. Risk management and position sizing
5. Deployment on cloud infrastructure
6. Mobile-responsive frontend interface
