# 01 – Data acquisition and exploratory analysis

## Data source

Initially I searched for publicly available datasets of customer feedback labelled with high‑level complaint categories.  Many relevant resources were found but were unsuitable for this project:

* **Consumer Complaint Database (U.S. CFPB):** The official complaint database maintained by the Consumer Financial Protection Bureau provides consumer complaints about financial products.  The description notes that the data consists of complaints about financial products and services, published after the company responds【56518301983924†L85-L95】.  However, the topics are financial (mortgages, loans, credit cards) rather than delivery, billing or app issues.  The licence information is also unclear【56518301983924†L97-L103】.  Therefore this dataset was not used.
* **Kaggle/GitHub datasets:** Several Kaggle and GitHub repositories advertise complaint classification datasets, but they either required accounts or returned errors in this environment.  Without the ability to authenticate, these sources could not be accessed.

Given these challenges I decided to generate a synthetic dataset for demonstration purposes.  A synthetic dataset has limitations—it cannot capture the full diversity of real customer language—but it enables the complete pipeline to be built and documented.

### Synthetic dataset generation

The dataset consists of 500 messages labelled across five categories:

| Category | Example description | Number of samples |
|---|---|---|
| **delivery_issue** | Complaints about late, missing, or damaged deliveries | 200 |
| **refund_request** | Requests for refunds or returns due to defective products or overcharges | 150 |
| **billing_problem** | Issues about unexpected fees, incorrect charges, or subscription costs | 80 |
| **app_bug** | Reports of crashes, errors or glitches in the mobile or web app | 50 |
| **other** | General questions, suggestions, or non‑complaint enquiries | 20 |

To create each message I combined a base template (e.g. “My package arrived late and the tracking information was inaccurate.” for `delivery_issue`) with optional extra phrases (e.g. “This is very frustrating”, “Please fix this issue”) using the Python `random` module.  This introduces minor variation without requiring an external language model.  The resulting CSV file (`data/raw/customer_feedback.csv`) contains two columns: `text` (the customer’s message) and `category` (the label).

The script used to generate the data is located in this repository’s history and can be reproduced.  Because the dataset is synthetic it is not bound by any external licence; it may be reused freely.

## Preliminary exploration

After regenerating the dataset to include a handful of ambiguous messages the file now contains 525 rows and two columns (`text` and `category`).  The updated class counts are:

| Category | Count |
|---|---|
| delivery_issue | 205 |
| refund_request | 155 |
| billing_problem | 85 |
| app_bug | 55 |
| other | 25 |

As designed, the classes remain imbalanced.  A bar chart of these counts is shown below:

![Class distribution]({{file:file-LvkftRsZM7dM9owFajKdEF}})

### Message lengths

To understand how long the messages are, I computed the number of characters per message.  The messages range from roughly 60 to 140 characters; most cluster between 90 and 120.  The histogram below shows the distribution:

![Message length distribution]({{file:file-HeXpYBdoZvNYioxjur6VJc}})

### Duplicates and empty texts

Because the synthetic data templates repeat phrases, many rows are duplicates.  Out of 525 records, 413 are exact duplicates and there are still no completely empty messages.  While duplicates artificially inflate the dataset size they also reflect common phrases customers use; for this exercise I will keep duplicates in the raw data but remove them during the preprocessing step for modelling to avoid overfitting.