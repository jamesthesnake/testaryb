# Mana Take-Home Test: AI Macro Specialist

## Notes

Completing the entire project is not required. We are looking for a demonstration of your skills and understanding of the problem.
Take approximately 4 hours to complete the project. 

We do not expect to complete the entire project in 4 hours. We have a point system for the project, and you can choose to complete the parts that you are most comfortable with.

Target to complete at least 100 points worth of work.

- Backend (Python)
  - Implement the retrieval system for the Macroeconomics Specialist (40 points)
  - Implement the q&a system for the Macroeconomics Specialist using OpenAI (40 points)
  - Implment citations for the data sources used in the economic analysis (20 points)
  - Implement the test cases for the Macroeconomics Specialist (20 points)

- Frontend (React or Streamlit)
    - Implement the chat interface for the Macroeconomics Specialist (40 points)
    - Implement the data visualization for the Macroeconomics Specialist (40 points)
    - Implement citations for the data sources used in the economic analysis (20 points)
    - Implement the test cases for the Macroeconomics Specialist (20 points)


## Project Overview

Create a full stack application that implements an AI-powered Macroeconomics Specialist, which is part of a larger collection of financial analyst AIs. 
This specialist should be capable of answering questions, generating reports, and providing insights on macroeconomic trends, indicators, and policies. 
The Macroeconomics Specialist should use a retrieval system to generate context from various economic data sources, then leverage OpenAI's API for analysis and insights. 

The application should use Python for the backend, React or Streamlit for the frontend, and integrate with OpenAI's API for 
AI capabilities.

## Key Requirements

1. Implement a MacroSpecialist class that manages the retrieval and analysis process for macroeconomic data and queries.
2. Create a robust retrieval system to generate context from various macroeconomic data sources.
3. Integrate with OpenAI's API for macroeconomic analysis, question-answering, and report generation.
4. Develop an interface for interacting with the Macroeconomics Specialist, displaying key economic indicators and allowing for query input.
5. Include comprehensive test coverage for both frontend and backend components.


## Sample Files and API KEYS

You have been provided with the API keys for OpenAI, PINECONE Serverless and FRED. 
Please use these keys to interact with the services. For Pinecone, please create an index 
called `takehome-macro-firstname-lastname` and use it to store the economic data embeddings 
for this project.

Sample data provided in /backend/data folder, and sample llm calls using instructor
model and openai are provided in /samples folder.


## Detailed Specifications

### Backend (Python)

1. MacroSpecialist Class:
   - Design and implement a MacroSpecialist class that orchestrates the retrieval and analysis process.

2. OpenAI Integration:
   - Integrate with OpenAI's API for macroeconomic analysis, question-answering.

3. Economic Data Integration:
   - Integrate with economic data provider APIs (e.g., FRED, World Bank, IMF) to fetch historical and current economic indicators.
     - For this exercise integrate with Fred API to get data - https://fred.stlouisfed.org/docs/api/fred/
     - Pull data for GDP, Inflation, Unemployment, Interest Rates, Exchange Rates

4. Economic Data Retrieval System:
   - Implement a vector database for storing and querying economic document embeddings.
   - Create indexing functions for various economic data types.
   - Develop a retrieval function that efficiently fetches relevant economic information based on user queries.

  - Questions the system should be able to answer
    - What is the current GDP of the US?
    - What is the inflation rate in the Eurozone?
    - What is the unemployment rate in Japan?
    - What is the current interest rate set by the Federal Reserve?
    - What is the current exchange rate between the US Dollar and the Euro?
    - Make a plot of the GDP growth rate in the US over the past 10 years.
    - Make a plot of US unemployment rate over the past 20 years.

5. Implement citations for the data sources used in the economic analysis.
  
6. Testing:
   - Write unit tests for the MacroeconomicsSpecialist class and its methods, including mock economic data.
   - Implement integration tests for the entire backend system, ensuring accurate economic data processing and analysis.

### Frontend (React)

1. Economic Dashboard:
   - Create a chat interface for querying the Macroeconomics Specialist about specific economic topics or indicators.
   - Should be able to display tables, plots and visualizations of economic data based on user queries.

2. Testing:
   - Write unit tests for individual components.
   - Implement integration tests for key user flows.


## Evaluation Criteria

1. System Design: Efficient and scalable architecture suitable for handling diverse economic data and analysis requests.
2. Code Quality: Clear, well-documented, and maintainable code, especially for the MacroeconomicsSpecialist class and economic data retrieval system.
3. Functionality: Accurate implementation of macroeconomic analysis features, including data retrieval, AI-powered insights, and report generation.
4. User Experience: Intuitive and responsive interface for economic data visualization and interaction with the Macroeconomics Specialist.
5. Testing: Comprehensive test coverage for both frontend and backend, with a focus on handling various economic scenarios and edge cases.
6. Documentation: Clear instructions for setting up and running the project, including any necessary steps for economic data ingestion and API key setup.

## Bonus Points

1. Implement a evaluation system that can measure how good the responses over time, and across different models.
2. Implement a system for the Macroeconomics Specialist to explain its reasoning and sources for its economic insights.

## Submission Guidelines

1. Submit a zip file containing:
   - Complete source code for both frontend and backend
   - All data files used for the project (e.g., economic datasets, pre-processed data)
   - A README.md file with:
     - Project overview
     - Setup instructions, including steps for economic data ingestion and any required API keys
     - API documentation, focusing on how to interact with the MacroeconomicsSpecialist class
     - Any assumptions or design decisions made, particularly around economic data handling and analysis
   - A brief (2-3 page) write-up discussing:
     - Your approach to designing the MacroeconomicsSpecialist class and economic data retrieval system
     - How you ensured the accuracy and reliability of economic analysis and predictions
     - Challenges faced in working with macroeconomic data and AI-powered analysis
     - Potential future improvements and expansions for a production-ready system

2. Ensure that the zip file is well-organized and includes all necessary files to run the project.

Good luck! We look forward to reviewing your submission and seeing your approach to AI-powered macroeconomic analysis.