# Phase 2: Candidate Generation & Blocking Report

## 1. Executive Summary & Iteration Comparison

| Iteration | Pair Recall Ceiling | Entity Recall Ceiling | Mean Cands/S1 | Median | p95 | Max | Missed Matches | Reduction Ratio | Runtime |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Iteration A (Baseline)** | **88.22%** | 70.30% | 30.89 | 35 | 50 | 50 | 201 | 99.9696% | 2.0s |
| **Iteration B (+Char & Combined TF-IDF)** | **96.72%** | 91.24% | 34.71 | 37 | 50 | 50 | 56 | 99.9659% | 12.2s |
| **Iteration C (Tuned Lean)** | **93.73%** | 83.76% | 24.36 | 30 | 30 | 30 | 107 | 99.9760% | 13.2s |

---

## 2. Segmented Breakdown (Best Configuration)

### A. By Country

| Country | S1 Count | True Pairs | Retained Pairs | Pair Recall Ceiling | Entity Recall Ceiling | Mean Candidates/S1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **India** | 200 | 665 | 617 | **92.78%** | 82.07% | 33.29 |
| **US** | 300 | 1,041 | 1,033 | **99.23%** | 97.18% | 35.65 |

### B. By Candidate Source (S2 vs S3)

| Source | True Pairs | Retained True Pairs | Recall Ceiling |
| :--- | :--- | :--- | :--- |
| **S2** | 842 | 825 | **97.98%** |
| **S3** | 864 | 825 | **95.49%** |

### C. Rule Attribution (True Matches Triggered by Rule)

| Blocking Rule Channel | True Matches Triggered |
| :--- | :--- |
| `rare_token` | 2,931 |
| `combined_tfidf` | 1,511 |
| `char_tfidf` | 1,366 |
| `exact_core_name` | 834 |
| `token_signature` | 680 |
| `house_number_block` | 680 |
| `exact_full_name` | 268 |
| `postal_code_block` | 87 |

---

## 3. Analysis of Top 20 Missed Matches

Below are the 20 most difficult true matches missed during candidate generation:

| S1 ID | Missed ID | Country | S1 Business Name | Match Business Name | Name Sim | Reason Missed |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `S1-846288553` | `S3-608375370` | India | Silver Classic Ventures Limited | সিলভার ক্লাসিক ভেঞ্চারস লিমিটেড | 9.7% | Severe alias/rebrand |
| `S1-872490692` | `S3-2466900` | India | Surya Estate Private Limited | સૂર્ય એસ્ટેટ પ્રાઇવેટ લિમિટેડ | 10.5% | Severe alias/rebrand |
| `S1-397244502` | `S2-752032264` | India | Om Products Pvt Ltd | ഓം പ്രൊഡക്ട്സ് പ്രൈവറ്റ് ലിമിറ്റഡ് | 11.3% | Severe alias/rebrand |
| `S1-461787088` | `S2-209661326` | India | Star Global Private Limited | स्टार ग्लोबल प्राइवेट लिमिटेड | 10.7% | Severe alias/rebrand |
| `S1-794237790` | `S3-297972667` | India | Aditya Consulting LLP | Avizeta | 28.6% | Spelling/transposition beyond floor |
| `S1-794237790` | `S2-410103751` | India | Aditya Consulting LLP | आदित्य कंसल्टिंग एलएलपी | 9.1% | Severe alias/rebrand |
| `S1-996631687` | `S3-870945428` | India | Jai Agro | जय एग्रो | 12.5% | Severe alias/rebrand |
| `S1-996631687` | `S3-437281640` | India | Jai Agro | जय एग्रो | 12.5% | Severe alias/rebrand |
| `S1-996631687` | `S3-885527791` | India | Jai Agro | जय एग्रो | 12.5% | Severe alias/rebrand |
| `S1-722214148` | `S3-230738506` | India | Maa Trading Private Limited | मां ट्रेडिंग प्राइवेट लिमिटेड | 10.7% | Severe alias/rebrand |
| `S1-894398468` | `S2-587537869` | India | Global Creative International Pvt Ltd | ഗ്ലോബൽ ക്രിയേറ്റീവ് ഇന്റർനാഷണൽ പ്രൈവറ്റ് ലിമിറ്റഡ് | 9.2% | Severe alias/rebrand |
| `S1-28372580` | `S2-666912858` | India | White Exports Private Limited | व्हाइट एक्सपोर्ट्स प्राइवेट लिमिटेड | 9.4% | Severe alias/rebrand |
| `S1-28372580` | `S3-671028499` | India | White Exports Private Limited | व्हाइट एक्सपोर्ट्स प्राइवेट लिमिटेड | 9.4% | Severe alias/rebrand |
| `S1-587709992` | `S3-526251906` | India | Classic Tech Private Limited | क्लासिक टेक प्राइवेट लिमिटेड | 10.7% | Severe alias/rebrand |
| `S1-134972377` | `S3-371516220` | India | Gujarat Urban Ventures Limited | Onyxbrix | 10.5% | Severe alias/rebrand |
| `S1-594326171` | `S3-511552500` | India | Laxmi East Products Private Limited | ಲಕ್ಷ್ಮಿ ಈಸ್ಟ್ ಪ್ರೊಡಕ್ಟ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್ | 10.3% | Severe alias/rebrand |
| `S1-594326171` | `S3-77695053` | India | Laxmi East Products Private Limited | ಲಕ್ಷ್ಮಿ ಈಸ್ಟ್ ಪ್ರೊಡಕ್ಟ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್ | 10.3% | Severe alias/rebrand |
| `S1-30180104` | `S2-479023055` | India | Southern Systems Private Limited | சதர்ன் சிஸ்டம்ஸ் பிரைவேட் லிமிடெட் | 9.1% | Severe alias/rebrand |
| `S1-829126346` | `S2-252791778` | India | Gujarat Properties Limited | गुजरात प्रॉपर्टीज लिमिटेड | 7.8% | Severe alias/rebrand |
| `S1-29246929` | `S2-260440562` | India | Vijay Healthcare Private Limited | বিজয় হেলথকেয়ার প্রাইভেট লিমিটেড | 9.2% | Severe alias/rebrand |


*Total diagnostic runtime: 163.2s*
