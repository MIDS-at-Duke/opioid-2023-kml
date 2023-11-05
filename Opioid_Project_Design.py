# Databricks notebook source
# MAGIC %md
# MAGIC # Opioid Project Design

# COMMAND ----------

# MAGIC %md
# MAGIC ### Question 1: What problem are you (or your stakeholder) trying to address? 
# MAGIC We are trying to address the increase in use and abuse of prescription opioids, leading to a rise in opioid addiction, a rise in prescription overdose deaths, and a rise in deaths from non-prescription opioids like heroin and fentanyl. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Question 2: What specific question are you seeking to answer with this project? 
# MAGIC
# MAGIC We will be estimating the effectiveness of policy interventions designed to limit the over-prescription of opioids using the following outcome variables: 
# MAGIC
# MAGIC - The volume of opioids prescribed 
# MAGIC - Drug overdose deaths 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Question 3: What is your hypothesized answer to your question? 
# MAGIC
# MAGIC We predict that an increase in opioid prescription drug regulation will decrease the volume of opioids prescribed but increase the number of drug overdose deaths. The rationale for this hypothesis is that non-prescription opioids with unregulated dosages and side effects will flood the market to fill the gap left from reduced prescription opioids. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Question 4: 
# MAGIC
# MAGIC *One of the hardest parts of developing a good data science project is developing a question that is actually answerable. Perhaps the best way to figure out if your question is answerable is to see if you can imagine what an answer to your question would look like. Below, draw the graph, regression table, etc. that you would consider to be an answer to your question. Then draw it again, so you have a model result for if your hypothesized answer is true, and a model result for if your hypothesized answer is false. (If the answer to your question is continuous, not discrete (e.g. “what is the level of inequality in the United States?”), draw it for high values (high inequality) and low values (low inequality)).*

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Result if your hypothesis is true: 

# COMMAND ----------

# Testing Connection to Azure Databricks
# Using compute power from azure databricks to process the data

