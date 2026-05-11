# Time-series data and application to stock markets

This project applies deep learning to time-series stock market data through a series of progressively more complex tasks, working with both Nasdaq (AAPL) and Vietnam stock datasets.

## Project Tasks

### Task 1 — Nasdaq stock price prediction (AAPL)

**Task 1.1 — Multi-feature extension:** Modified the demo code to use all features of the data instead of just one. Used `.values` to grab all the features, predicting the close price instead of the open. Kept most of the demo code normalization but treated volume separately.

**Task 1.2 — Nth day forecast:** Gave the model the capability to predict the price of the kth day in the future. Reduced the size of each window by N to make sure we don't go out of bounds, then changed the label to be the Nth day ahead rather than the next one.

**Task 1.3 — K consecutive days forecast:** Changed window size and the label by variable k. Used a loop to add each date up to k. The final dense layer of the model was changed to have K nodes representing each predicted day.

### Task 2 — Vietnam stock price prediction (VIC)

Selected VIC-VNINDEX, the parent company of all of the Vin Group subsidiaries, one of the largest companies in Vietnam and very diverse in the markets it participates in.

Evaluated the auxiliary data (Finance, Industry, Dividend) but determined the number of data points was not enough to provide relevant information to the model. Stuck to the same process used for Task 1, replicating tasks 2.1, 2.2, and 2.3 for the VIC stock. Increased the number of epochs to account for VIC being a more volatile stock living in a more speculative market.

### Task 3 — Trading signal identification

Transformed the task from predicting a number to classifying a Buy or a Sell feature. Used the 5 existing features (Open, High, Low, Close, Volume) to calculate 5 new features: SMA_20, SMA_50, RSI, MACD, MACD_signal — classic ways that stock traders have identified when it is a good time to buy a stock.

Defined a threshold and window size parameter to determine what is considered a buy or a sell given continued positive or negative movement. Changed the head of the models to give buy or sell classification.

**Note:** The model didn't predict a single buy, only "don't buy". Tried class weighting, tweaking the horizon and threshold, switching to LSTM/GRU, and even a simpler model — none worked. Concluded that it is not feasible to create a model that can accurately predict when to buy because if there was then everyone would be doing it.

### Task 4 — Portfolio composition for Vietnam market

**Task 4.1 — Profitable stock selection:** Selected 10 profitable stocks with long trading history from diverse markets:

| Ticker | Exchange   | Notes                  |
| ------ | ---------- | ---------------------- |
| AGF    | UpcomIndex | Aquaculture            |
| BCC    | HNXIndex   | Cement                 |
| BT6    | UpcomIndex | Construction materials |
| HAP    | VNINDEX    | Trading                |
| KHA    | UpcomIndex | Construction           |
| LAF    | VNINDEX    | Agricultural           |
| REE    | VNINDEX    | Electrical engineering |
| SAM    | VNINDEX    | Cable & telecom        |
| VIC    | VNINDEX    | Real estate (Vingroup) |
| VNR    | HNXIndex   | Reinsurance            |
| VTL    | HNXIndex   | Travel & tourism       |

Implemented the same concept from Task 1.3 but with a much larger K value of 60 days. Used glob to load all stocks into a dictionary and filtered them by year. As opposed to 1.3 that was trained on a single stock, this model was trained by pooling all the data from each stock, which was around 28,000 windows.

**Task 4.2 — Risk management:** Employed the same model but measured volatility to decide which stocks seem to be a risky investment. Volatility measures how much a stock tends to fluctuate — stocks with high volatility are risky because you could go through periods of negative growth unexpectedly. Computed as the standard deviation of future returns predictions, multiplied by sqrt(252) to annualize since there are 252 trading days per year and the convention in finance is to express volatility on an annual basis. Flagged the top three companies with the highest volatility as too risky.

**Task 4.3 — Portfolio composition:** Combined both metrics into an optimal portfolio construction strategy. Used the formula `score = α × predicted_return - β × predicted_volatility` where alpha and beta are tunable variables to determine the strategy. Determined three different types of investors — balanced, aggressive, and conservative — and created different weight values to represent these positions. Visualized the diversification of the portfolios in a bar chart to see what % each stock makes up of them.

## Setup and Running Instructions

### Prerequisites
- Google Colab account
- Google Drive with the following folder structure:
  - `/MyDrive/selected_stocks/AAPL.csv` (for Task 1)
  - `/MyDrive/selected_stocks/VIC-VNINDEX-History.csv` (for Tasks 2 and 3)
  - `/MyDrive/selected_stocks/task4_stocks/` containing all 10 Vietnam stock CSVs (for Task 4)

### Running the Notebook

1. Open the project notebook in Google Colab
2. Mount your Google Drive when prompted
3. Run cells top to bottom — the notebook is organized as one Colab notebook with shared functions defined for normalization, denormalization, and building the actual model, so each task can be included without rewriting these sections
4. Each task section can be run independently after the Setup and Helper Functions sections at the top

### Notebook Structure

- **Setup** — imports, Drive mounting, shared parameters
- **Helper functions** — shared across all tasks (normalization, denormalization, model building)
- **Task 1.1, 1.2, 1.3** — AAPL stock prediction variants
- **Task 1 Comparison** — MSE/RMSE comparison table
- **Task 2.1, 2.2, 2.3** — VIC stock prediction variants
- **Task 2 Comparison** — MSE/RMSE comparison table
- **Task 3** — Buy/Sell signal classification
- **Task 4.1, 4.2, 4.3** — Portfolio construction across 10 Vietnam stocks
