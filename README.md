# Tweet Popularity Predictor

A comprehensive machine learning pipeline that performs three distinct tasks on social media data: emotion classification, hashtag generation, and popularity prediction with enterprise data warehouse integration.

## 🚀 Features

- **Emotion Classification**: Classify emotions in tweets using DistilBERT
- **Hashtag Generation**: Generate relevant hashtags using GPT-2
- **Popularity Prediction**: Predict tweet popularity using embeddings and regression
- **Multi-Task Processing**: Run all tasks simultaneously or individually
- **Data Warehouse Integration**: Automated Snowflake storage for scalable analytics
- **Command-Line Interface**: Easy-to-use CLI for batch processing or single predictions
- **Interactive Dashboards**: Business intelligence visualization of ML outputs

## 📋 Requirements

```bash
pip install pandas torch transformers scikit-learn numpy argparse snowflake-connector-python
```

### Model Dependencies
- `bhadresh-savani/distilbert-base-uncased-emotion` (Emotion Classification)
- `gpt2` (Hashtag Generation)  
- `distilbert-base-uncased` (Text Embeddings for Popularity Prediction)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tweet-popularity-predictor.git
cd tweet-popularity-predictor
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables for Snowflake (optional):
```bash
cp .env.template .env
# Edit .env with your Snowflake credentials
```

## 💻 Usage

### ML Pipeline

#### Command Line Arguments

| Argument | Type | Description | Required |
|----------|------|-------------|----------|
| `--file` | string | Path to input CSV file | Optional* |
| `--content` | string | Tweet content for single prediction | Optional* |
| `--number_of_shares` | int | Number of shares for the tweet | Optional* |
| `--number_of_likes` | int | Number of likes for the tweet | Optional* |
| `--task` | string | Task to perform: `emotion`, `hashtags`, `popularity`, `all` | **Required** |

*Either `--file` OR individual tweet parameters must be provided.

#### Examples

##### 1. Complete Pipeline - All Tasks
```bash
python mtml_model.py --file tweets.csv --task all
```

##### 2. Single Tweet Analysis
```bash
python mtml_model.py --content "I'm feeling great today!" --number_of_shares 10 --number_of_likes 25 --task emotion
```

### Data Warehouse Operations

#### Snowflake Integration
```bash
# Set up complete Snowflake infrastructure and load data
python snowflake_ops.py --all --csv_path final_result.csv

# Individual operations
python snowflake_ops.py --create_db --create_schema --create_warehouse
python snowflake_ops.py --create_table --csv_path final_result.csv
```

#### Complete End-to-End Workflow
```bash
# 1. Run ML predictions
python mtml_model.py --file tweets.csv --task all

# 2. Load results to Snowflake  
python snowflake_ops.py --all --csv_path final_result.csv

# 3. Results are now queryable in Snowflake for dashboard creation
```

## 📊 Input Data Format

### CSV File Requirements
Your input CSV file should contain the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `content` | string | Tweet/post content |
| `number_of_shares` | int | Number of times shared |
| `number_of_likes` | int | Number of likes received |

Example CSV:
```csv
content,number_of_shares,number_of_likes
"I love this new restaurant!",15,42
"Traffic is terrible today",3,8
"Just finished my workout",7,23
```

## 📤 Output

### ML Pipeline Output
The model generates a `final_result.csv` file containing:
- **Original columns**: `content`, `number_of_shares`, `number_of_likes`
- **Emotion**: Predicted emotion label
- **Hashtags**: Generated hashtag text
- **hashtags_final**: Extracted hashtags as list
- **popularity**: Popularity score (0-100 scale)
- **Additional features**: `content_length`, `hashtags_count`

### Snowflake Database Schema
```sql
CREATE TABLE PREDICTED_TWEETS (
    AUTHOR STRING,
    CONTENT STRING,
    COUNTRY STRING,
    DATE_TIME STRING,
    ID INT,
    LANGUAGE STRING,
    LATITUDE FLOAT,
    LONGITUDE FLOAT,
    NUMBER_OF_LIKES INT,
    NUMBER_OF_SHARES INT,
    EMOTION STRING,
    HASHTAGS STRING,
    HASHTAGS_FINAL ARRAY,
    CONTENT_LENGTH INT,
    HASHTAGS_COUNT INT,
    POPULARITY FLOAT
);
```

## 🏗️ System Architecture

### ML Processing Pipeline
```
Raw CSV → Multi-Task ML Models → Predictions → CSV Output
    ↓              ↓                    ↓           ↓
Input Data → [Emotion|Hashtag|Popularity] → Structured Results
```

### Data Architecture Flow
```
ML Predictions → Snowflake Data Warehouse → Business Intelligence Dashboards
     ↓                      ↓                           ↓
 final_result.csv → PREDICTED_TWEETS table → Analytics & Insights
```

### Task-Specific Architecture

#### Task 1: Emotion Classification
- **Model**: DistilBERT (fine-tuned on emotion data)
- **Output**: Emotion labels (joy, sadness, anger, fear, surprise, love)
- **Method**: Transformer-based text classification

#### Task 2: Hashtag Generation  
- **Model**: GPT-2
- **Output**: Generated text with hashtags
- **Method**: Autoregressive text generation with regex extraction

#### Task 3: Popularity Prediction
- **Models**: DistilBERT (embeddings) + Linear Regression
- **Features**: 
  - Text embeddings (768-dim)
  - Content length
  - Hashtag count  
  - Number of shares/likes
- **Output**: Popularity score (0-100)

## 📊 Dashboard Results & Insights

### Emotion Analysis Over Time
![Emotion Timeline](results/dashboards/emotion_popularity_timeline.png)

**Key Findings:**
- Joy is the dominant emotion (7,746 instances vs 1,377 anger)
- Clear temporal patterns show emotion spikes during major events (2015-2017)
- Model successfully captures emotional trends in social media data

### Top Hashtags & Emotions Analysis
![Hashtag Analysis](results/dashboards/hashtags_emotions_summary.png)

**Business Insights:**
- Political hashtags (#ActOnClimate, #TS1989) drive highest engagement (329, 229 posts)
- Joy accounts for 77% of high-popularity content
- Hashtag generation model identifies trending topics effectively

## 🔧 Advanced Features

### Model Persistence
```python
# Export trained popularity model
task3.export_model('popularity_model.pkl')

# Load pre-trained model
task3.load_model('popularity_model.pkl')
```

### Data Processing Capabilities
- **Batch Processing**: Up to 10,000 rows per execution
- **Preprocessing Pipeline**: Automatic data cleaning and feature engineering
- **Array Handling**: Proper Snowflake ARRAY type conversion for hashtags
- **Schema Validation**: Data quality checks before warehouse loading

### Scalability Features
- **Cloud Storage**: Snowflake integration for enterprise-scale analytics
- **SQL Queryable**: All ML outputs accessible via standard SQL
- **Concurrent Processing**: Multi-task execution optimization
- **Memory Management**: Efficient tensor processing for large datasets

## ⚡ Performance Benchmarks

| Task | Processing Speed | Model Size |
|------|------------------|------------|
| **Emotion Classification** | ~100-500 tweets/second | ~250MB |
| **Hashtag Generation** | ~50-200 tweets/second | ~500MB |  
| **Popularity Prediction** | ~200-800 tweets/second | ~250MB |
| **All Tasks Combined** | ~30-100 tweets/second | ~1GB total |

*Performance varies based on hardware configuration and input complexity.*

## 🚨 Important Requirements

1. **GPU Recommended**: For optimal transformer model performance
2. **Memory**: Minimum 8GB RAM for full pipeline execution
3. **Storage**: ~1GB for model downloads on first run
4. **Snowflake Account**: Required for data warehouse functionality
5. **Internet**: Required for initial model downloads from Hugging Face

## 🐛 Troubleshooting

### Common Issues & Solutions:

| Issue | Solution |
|-------|----------|
| **CUDA/GPU errors** | Install PyTorch with appropriate CUDA version |
| **Memory errors** | Reduce batch size or process smaller files |
| **Model download timeout** | Check internet connection and Hugging Face status |
| **Snowflake connection failed** | Verify credentials in environment variables |
| **Import errors** | Ensure all requirements are installed: `pip install -r requirements.txt` |

### Performance Optimization:
- Enable GPU acceleration if available
- Process data in smaller chunks for limited memory systems
- Pre-download models for offline environments
- Use Snowflake's COPY command for large dataset uploads

## 🤝 Contributing

Contributions are welcome! Please feel free to submit:
- Bug reports and fixes
- Feature requests and implementations  
- Performance improvements
- Documentation enhancements

## 📄 License

This project uses pre-trained models with their respective licenses. Please review individual model licenses before commercial deployment.

## 🙏 Acknowledgments

- Hugging Face for transformer models
- Snowflake for data warehouse capabilities
- Open source ML community for foundational tools
