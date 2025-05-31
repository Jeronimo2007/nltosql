from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from datasets import load_dataset
from dotenv import load_dotenv
import re
import os

load_dotenv() 
def reduce_context(prompt: str, sql_context: str, max_tables: int = 20) -> str:
    """
    Return up to max_tables CREATE TABLE (and any inline INSERT) blocks from sql_context
    that are most relevant to prompt. Always include the 'bus_routes' table if present.
    """
    prompt_lower = prompt.lower()

    
    table_defs = re.findall(
        r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([\w\.]+)',
        sql_context,
        flags=re.IGNORECASE
    )
    ordered_tables = []
    for tbl in table_defs:
        if tbl not in ordered_tables:
            ordered_tables.append(tbl)

    
    scored = []
    for full_name in ordered_tables:
        short_name = full_name.split('.')[-1]
        tokens = [tok for tok in re.split(r'[_\W]+', short_name.lower()) if tok]
        score = sum(tok in prompt_lower for tok in tokens)
        if score > 0:
            scored.append((full_name, score))

    
    scored.sort(key=lambda x: x[1], reverse=True)
    selected = [name for name, _ in scored[:max_tables]]


    
    for name in ordered_tables:
        if len(selected) >= max_tables:
            break
        if name not in selected:
            selected.append(name)

    
    blocks = re.split(r'(?=CREATE\s+TABLE)', sql_context, flags=re.IGNORECASE)
    blocks = [blk.strip() for blk in blocks if blk.strip()]

    
    selected_blocks = []
    for table in selected:
        pattern = rf'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?{re.escape(table)}'
        for blk in blocks:
            if re.search(pattern, blk, flags=re.IGNORECASE):
                selected_blocks.append(blk)
                break

    
    seen = set()
    unique_blocks = []
    for blk in selected_blocks:
        if blk not in seen:
            seen.add(blk)
            unique_blocks.append(blk)

    # 9) Return as a single string
    return "\n\n".join(unique_blocks)


user_prompt ="How many marine species are found in the Southern Ocean?"


dataset = load_dataset("gretelai/synthetic_text_to_sql")


column = dataset['train'].to_pandas()['sql_context']

column_context = "\n".join(column.astype(str).to_list())


result = reduce_context(user_prompt, column_context)






system_message = """
You are an expert SQL assistant specialized in natural language to SQL conversion. Follow these guidelines:

1. Schema Analysis:
   - Identify all relevant tables and their relationships
   - Note primary and foreign keys
   - Consider table aliases for complex queries

2. Query Planning:
   - Determine the required columns and tables
   - Identify necessary joins
   - Plan appropriate aggregations
   - Consider performance implications

3. Query Generation Rules:
   - Use proper SQL syntax and best practices
   - Include appropriate table aliases
   - Add meaningful column aliases
   - Use proper date/time functions
   - Handle NULL values appropriately
   - Consider edge cases

4. Output Format:
   - Generate ONLY the SQL query
   - Use consistent formatting
   - Include comments for complex logic
   - Ensure the query is executable

Remember to:
- Consider data types when comparing values
- Use appropriate indexing hints if needed
- Handle potential NULL values
- Consider query performance
- Use proper SQL standards
"""

CoT = """
Example 1: Basic Aggregation with Filtering
User: How many marine species are found in the Southern Ocean?

SQL Schema Context:
CREATE TABLE Ocean(
    id INT PRIMARY KEY,
    region VARCHAR(20),
    temperature DECIMAL(5,2)
);
CREATE TABLE SpeciesWaterTemp(
    SpeciesID INT,
    Date DATE,
    WaterTemp FLOAT,
    FOREIGN KEY (SpeciesID) REFERENCES marine_species(id)
);
CREATE TABLE marine_species(
    id INT PRIMARY KEY,
    name VARCHAR(50),
    common_name VARCHAR(50),
    location VARCHAR(50),
    conservation_status VARCHAR(20)
);

Assistant:
SELECT COUNT(*) as species_count
FROM marine_species
WHERE location = 'Southern Ocean';

---------------------
Example 2: Complex Join with Aggregation
User: What is the average temperature and number of species for each ocean region?

SQL Schema Context:
CREATE TABLE Ocean(
    id INT PRIMARY KEY,
    region VARCHAR(20),
    temperature DECIMAL(5,2)
);
CREATE TABLE marine_species(
    id INT PRIMARY KEY,
    name VARCHAR(50),
    location VARCHAR(50)
);

Assistant:
SELECT 
    o.region,
    AVG(o.temperature) as avg_temperature,
    COUNT(DISTINCT ms.id) as species_count
FROM Ocean o
LEFT JOIN marine_species ms ON o.region = ms.location
GROUP BY o.region
ORDER BY avg_temperature DESC;

---------------------
Example 3: Date-Based Analysis with Multiple Conditions
User: Show the top 5 species with the highest temperature readings in February 2023

SQL Schema Context:
CREATE TABLE SpeciesWaterTemp(
    SpeciesID INT,
    Date DATE,
    WaterTemp FLOAT
);
CREATE TABLE marine_species(
    id INT PRIMARY KEY,
    name VARCHAR(50),
    common_name VARCHAR(50)
);

Assistant:
SELECT 
    ms.name,
    ms.common_name,
    MAX(st.WaterTemp) as max_temperature
FROM SpeciesWaterTemp st
JOIN marine_species ms ON st.SpeciesID = ms.id
WHERE st.Date BETWEEN '2023-02-01' AND '2023-02-28'
GROUP BY ms.id, ms.name, ms.common_name
ORDER BY max_temperature DESC
LIMIT 5;

---------------------
Example 4: Complex Aggregation with Window Functions
User: For each ocean region, show the temperature difference from the regional average

SQL Schema Context:
CREATE TABLE Ocean(
    id INT PRIMARY KEY,
    region VARCHAR(20),
    temperature DECIMAL(5,2),
    measurement_date DATE
);

Assistant:
SELECT 
    region,
    temperature,
    measurement_date,
    temperature - AVG(temperature) OVER (PARTITION BY region) as temp_difference
FROM Ocean
ORDER BY region, measurement_date;
"""


user = """
Given the following examples and context:

Examples:
{examples}

Please generate a SQL query that:
1. Accurately answers the user's question
2. Uses proper SQL syntax and best practices
3. Considers performance implications
4. Handles edge cases appropriately

SQL Schema Context:
{context}

User Question: {user_prompt}

Remember to:
- Use appropriate table aliases
- Include meaningful column aliases
- Handle NULL values
- Consider data types
- Use proper indexing where beneficial
"""


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_message),
        ("user", user),
    ]
)


filtering_examples = """
Example 1: Complex Business Logic
User Prompt: Calculate the percentage of defense contracts awarded to companies in California, grouped by contract amount ranges

SQL Context:
CREATE TABLE contract_states (
    company_name TEXT,
    contract_id INT PRIMARY KEY,
    state TEXT,
    industry_category VARCHAR(50)
);

CREATE TABLE defense_contracts (
    contract_id INT PRIMARY KEY,
    contract_amount FLOAT,
    award_date DATE,
    contract_type VARCHAR(50),
    FOREIGN KEY (contract_id) REFERENCES contract_states(contract_id)
);

Assistant: 
CREATE TABLE contract_states (
    company_name TEXT,
    contract_id INT PRIMARY KEY,
    state TEXT,
    industry_category VARCHAR(50)
);
CREATE TABLE defense_contracts (
    contract_id INT PRIMARY KEY,
    contract_amount FLOAT,
    award_date DATE,
    contract_type VARCHAR(50),
    FOREIGN KEY (contract_id) REFERENCES contract_states(contract_id)
);

-----------------
Example 2: Time-Series Analysis
User Prompt: Show the monthly trend of equipment maintenance frequency and costs

SQL Context:
CREATE TABLE equipment_maintenance(
    equipment_id INT PRIMARY KEY,
    equipment_type TEXT,
    maintenance_frequency INT,
    last_inspection DATE,
    maintenance_cost DECIMAL(10,2)
);

CREATE TABLE maintenance_history(
    maintenance_id INT PRIMARY KEY,
    equipment_id INT,
    maintenance_date DATE,
    cost DECIMAL(10,2),
    FOREIGN KEY (equipment_id) REFERENCES equipment_maintenance(equipment_id)
);

Assistant:
CREATE TABLE equipment_maintenance(
    equipment_id INT PRIMARY KEY,
    equipment_type TEXT,
    maintenance_frequency INT,
    last_inspection DATE,
    maintenance_cost DECIMAL(10,2)
);
CREATE TABLE maintenance_history(
    maintenance_id INT PRIMARY KEY,
    equipment_id INT,
    maintenance_date DATE,
    cost DECIMAL(10,2),
    FOREIGN KEY (equipment_id) REFERENCES equipment_maintenance(equipment_id)
);
"""


filtering_template = """
{examples}

Given the examples above, your task is to:
1. Look at the examples and find the table structure that matches the user's question
2. Pay special attention to fields mentioned in the user's question (like dates, names, etc.)
3. Return EXACTLY the same table structure as shown in the examples, including any INSERT statements if present
4. Do not create new table structures or modify the existing ones
5. Only return the table structure and its data, nothing else

User Prompt: {input}

Context: {context}

Remember: 
- Look for tables that contain ALL the fields mentioned in the user's question
- For date-related queries, find tables that have date fields
- For name-related queries, find tables that have name fields
- For country/location queries, find tables that have country/location fields
- The table MUST have all the fields needed for the operation (e.g., for an UPDATE with dates, it must have date fields)
- Return ONLY the exact table structure and ONLY ONE from the examples that matches these criteria, including any INSERT statements
- If you find a table with all the required fields, use that one even if its name is slightly different
"""


filtering_prompt = ChatPromptTemplate.from_template(filtering_template)


llm_filtering = RunnablePassthrough.assign(
    input=lambda x: user_prompt,
    examples=lambda x: filtering_examples,
    context = lambda x: x["context"]
    ) | filtering_prompt | ChatOpenAI(
    model_name="deepseek/deepseek-chat-v3-0324:free",
    openai_api_key=os.getenv("api_key"),
    openai_api_base=os.getenv("base_url"),
) | StrOutputParser()


# First get the filtered context
filtered_context = llm_filtering.invoke({
    "input": user_prompt,
    "context": reduce_context(user_prompt, column_context)
})


chain = RunnablePassthrough.assign(
    examples=lambda x: CoT,
    context=lambda x: filtered_context,  
    user_prompt=lambda x: user_prompt
) | prompt | ChatOpenAI(
    model_name="meta-llama/llama-3.3-8b-instruct:free",
    openai_api_key=os.getenv("api_key"),
    openai_api_base=os.getenv("base_url"), 
) | StrOutputParser()


result = chain.invoke({"input": user_prompt})

if __name__ == "__main__":
    print(reduce_context(user_prompt, column_context))
    print("============================================================")
    print(filtered_context)
    print("============================================================")
    print(result)
