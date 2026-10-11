# Northwind QnA — Test Questions

Questions to exercise the multi-agent Northwind chatbot
(`python examples/northwind_qna.py`).

Data: 200 orders (July 1996 – February 1997), 20 products, 20 customers,
9 employees, 8 categories. Read-only OData service.

## Single-shot questions

1. **Multi-condition + date math**
   > How many orders were shipped to Germany in Q1 1997, how many were delivered within 3 weeks of the order date, and which ones took longer?

2. **Grouping + aggregation**
   > Which country received the most orders in 1997, and what was the average freight cost for orders to that country?

3. **Date range + threshold**
   > In the first half of 1997, how many orders had a freight cost over $100, and what was the total freight for those orders?

4. **Category analysis (join Products → Categories)**
   > For each product category, what is the total inventory value (unit price × units in stock) and which category has the highest total?

5. **Customer ranking**
   > Which 5 customers placed the most orders in 1997, and what was each one's total order count?

6. **Discrepancy detection (date comparison)**
   > How many orders in 1997 were shipped more than 5 days after the required delivery date, and list the worst 3 offenders?

7. **Product + category join with filter**
   > How many products in the "Beverages" category are currently out of stock (units in stock = 0), and what are their names?

8. **Distribution (order vs ship gap)**
   > Across all 1997 orders, what is the median number of days between order date and ship date, and how many orders shipped in under a week?

9. **Multi-entity cross (customers + countries + orders)**
   > Which country has the most customers, and for that country, what is the total number of orders placed by its customers in 1997?


## Multi-turn follow-up sequences

Ask these in order (the shared state persists between turns):

**Sequence A — Drill down on Germany**
1. How many orders were delivered to Germany?
2. Of those, how many had freight over 100?
3. What was the total freight for all German orders?
4. Which German customer had the highest total freight?

**Sequence B — Time-based drill**
1. How many orders were placed in Q1 1997?
2. Of those, how many were shipped to France?
3. What was the average freight for French orders in Q1 1997?

**Sequence C — Freight analysis drill**
1. What is the total freight spent across all 1997 orders?
2. What percentage of that total came from orders shipped to the USA?
3. What was the highest single freight cost in 1997, and which customer did it belong to?

**Sequence D — Product inventory drill**
1. What is the total inventory value of all products (unit price × units in stock)?
2. How many products have an inventory value above $2,000?
3. Of those, which product has the highest inventory value and what is it?


## Notes

- **Joins** (questions 4, 7, 9) require the agent to reconcile two datasets on
  IDs. The agent fetches each entity in separate calls; only the last fetch
  persists in shared state, so the analyst works with one dataset at a time.
- **Date math** (questions 1, 6, 8) stress-tests the `analyze_data` tool on
  large result sets.
- **Multi-turn** sequences test that `shared_state` persists and the analyst
  can re-query the same data with different Python code.
- The data spans July 1996 to February 1997. "Last quarter" maps to Q1 1997
  per the system prompt.
- Type `reset` between unrelated sequences to clear shared state.
