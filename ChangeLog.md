# Project 1: Data Cleaning & Preparation Change Log

## Change ID: CR001
- **Description:** Imputed numeric columns using median values and rounded float columns to 2 decimal places. Standardized text columns (City) to title case and removed whitespace. Checked and formatted 'Timestamp' column to datetime.
- **Impact:** Preserved records by resolving null values and standardized date formats.
- **Status:** Resolved

## Change ID: CR002
- **Description:** Handled duplicate records based on `Order_ID` by stripping whitespaces and keeping the first occurrence.
- **Impact:** Cleaned 1 initial duplicate record. Current duplicate rate is 0%.
- **Status:** Resolved