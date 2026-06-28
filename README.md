# Data Copilot - AI-Powered Data Analysis Platform

## Purpose

- Combines conversational AI with autonomous data exploration
- Uses CoTAS (Chain-of-Thought-Action-Search) methodology
- Users can:
  - Upload datasets (CSV, Excel, SQLite, JSON)
  - Chat with AI about their data and goals
  - Trigger autonomous iterative analysis (think, plan, code, execute)
  - View live reasoning and execution via CoTAS streaming insights
- Ideal for data analysts, researchers, business users

##  Tech Stack

### Backend
- **Flask** – API framework
- **Python 3.8+** – Core programming language
- **Google Gemini API (gemini-2.5-flash)** – LLM for orchestration
- **Pandas** – Data processing
- **SQLite3** – Handles database files

### Frontend
- **Vanilla JavaScript** – Interactive logic
- **HTML5 & CSS3** – Responsive design
- **Server-Sent Events (SSE)** – Real-time updates

### Data Processing
- Supports file parsing: CSV, Excel, SQLite, JSON
- Executes Python scripts in a sandboxed environment
- Uses session-based storage for state management
- **Data science stack** (pinned in `requirements.txt`): matplotlib, seaborn, scipy, scikit-learn, statsmodels, plotly (plus numpy/pandas)

Run import tests after installing dependencies:

```bash
python -m unittest discover -s tests -v
```

### Installation Steps 1. **Clone the repository**
bash
   git clone https://github.com/Rex-8/ConCore ConCore 
   cd ConCore
2. **Create a virtual environment**
bash
   python -m venv CCenv
   source CCenv/bin/activate  # On Windows: CCenv\Scripts\activate
3. **Install dependencies**
bash
    pip install -r requirements.txt
4. **Configure environment variables** Create a .env file in the root directory:
env
   GOOGLE_API_KEY=your_gemini_api_key_here
5. **Run the application**
bash
   python app.py
6. **Access the application** Open your browser and navigate to:
http://localhost:5000

##  How to Use 
1. **Create Session** - Click "Create New Session" to initialize your workspace 
2. **Upload Data** - Select and upload your dataset (CSV, Excel, SQLite, or JSON) 
3. **Chat** - Provide context about your data, goals, or ask questions 
4. **Generate Insights** - Click "Run CoTAS Analysis" to start autonomous data exploration 
5. **Review Results** - Watch the AI agent think, code, and analyze in real-time

## Testing To verify the installation: 
1. Start the server with python app.py 
2. Create a session through the UI 
3. Upload a sample CSV file 
4. Send a test message in chat 
5. Trigger insight generation and observe the streaming output

##  Maintainers - 
** [@Rex-8](https://github.com/Rex-8) **

## Contributing 
We welcome contributions! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on:
 - Project architecture
 - Pull request requirements
 - Code standards
 - Testing requirements

## Links
 - [Contributing Guidelines](CONTRIBUTING.md)
 - [Issue Tracker](https://github.com/Rex-8/ConCore/issues)

## Important Notes
 - The .env file containing your API key should **never** be committed to version control
 - Session data is stored locally in the storage/ directory
 - Python scripts are executed in a subprocess with timeout protection
 - Maximum file upload size is 100MB

## Features 
 - Multi-format data ingestion (CSV, Excel, SQLite, JSON)
 - AI-powered metadata extraction
 - Conversational context management
 - Autonomous CoTAS analysis loops
 - Real-time streaming insights
 - Safe sandboxed code execution
 - Session-based data isolation
 - Chat history tracking