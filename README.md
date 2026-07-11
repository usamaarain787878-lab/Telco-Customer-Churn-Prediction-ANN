# DecodeLabs Industrial Internship - Data Integrity Pipeline

## Project 1: Data Cleaning & Preparation
**Batch:** 2026 | **Developer:** Usama Arain

### Project Overview
This repository contains the production-grade data integrity pipeline built during the DecodeLabs Industrial Internship. The objective of this project is to clean, transform, and sanitize raw transaction datasets into a reliable "Gold Standard" source of truth before any analytical modeling.

### Key Pipeline Features
- **Column Standardization:** Stripped whitespaces and corrected formatting issues across text features (e.g., City converted to Title Case).
- **Strategic Imputation:** Handled missing values using statistical metrics (Median for numeric and Mode for categorical columns) without dropping valid rows.
- **Data Formatting:** Standardized Float columns to 2 decimal places and formatted 'Timestamp' values strictly into proper datetime series.
- **Duplicates Management:** Removed inflated transaction counts and ensured a 0% error rate based on unique `Order_ID` tracking.

### Quality Matrix Report Output
- **Initial Transformed Duplicates Cleaned:** 1
- **Total Inflated Record Duplicates Found:** 0
- **Total Missing/Null Gaps Remaining:** 0

### How to Run
```bash
python clean_data.py