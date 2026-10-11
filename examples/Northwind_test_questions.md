# Northwind QnA — Test Questions

Questions to exercise the multi-agent Northwind chatbot
(`python -m miniagent.northwind_qna`).

They cover multi-condition filters, date arithmetic, joins, grouping,
multi-step analysis, and multi-turn follow-ups.

## Single-shot questions

1. **Multi-condition + date math**
   > How many orders were shipped to Germany in Q2 1997, how many were delivered within 3 weeks of the order date, and which ones took longer?

2. **Grouping + aggregation**
   > Which country received the most orders in 1997, and what was the average freight cost for orders to that country?

3. **Date range + threshold**
   > In the first half of 1997, how many orders had a freight cost over $100, and what was the total freight for those orders?

4. **Category analysis (join Products → Categories)**
   > For each product category, what is the total revenue (unit price × units in stock) and which category has the highest total?

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

10. **Compound business metric**
    > For the top 10 most expensive products, how many of them had any orders in 1997, and what was the total revenue from those orders?

## Multi-turn follow-up sequences

Ask these in order (the shared state persists between turns):

**Sequence A — Drill down on Germany**
1. How many orders were delivered to Germany?
2. Of those, how many had freight over 100?
3. What was the total freight for all German orders?
4. Which German customer had the highest total freight?

**Sequence B — Time-based drill**
1. How many orders were placed in Q2 1997?
2. Of those, how many were shipped to France?
3. What was the average freight for French orders in Q2 1997?

**Sequence C — Product investigation**
1. How many products are in the Beverages category?
2. Which of those has the highest unit price?
3. How many orders included that product?

## Notes

- **Joins** (questions 4, 7, 9, 10) are the hardest — the agent must reconcile
  two datasets on IDs. If it struggles, nudge it: *"Use the ProductID/CategoryID
  to join them."*
- **Date math** (questions 1, 6, 8) stress-tests the `analyze_data` tool on
  large result sets.
- **Multi-turn** sequences test that `shared_state` persists and the analyst
  can re-query the same data with different Python code.
- "1997" is unambiguous; "last quarter" maps to Q2 1997 per the system prompt.
- Type `reset` between unrelated sequences to clear shared state.

## How to run

```bash
cd ~/Documents/Avi-Work
python -m miniagent.examples.northwind_qna
```

Paste one question at a time. `quit`/`exit`/`q` or Ctrl-C to stop. `reset` clears context.
