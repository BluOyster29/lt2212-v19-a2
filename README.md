#

LT2212 V19 Assignment 2
From Asad Sayeed's statistical NLP course at the University of Gothenburg.

My name: Robert Rhys Thomas 

Additional instructions
At the moment simdoc is all automated, I have removed the argparse just for convenience. All the vector files are attached in the zip file. I have some issues with cosine simularity and I am still working on this. 

To Do
- Doc Strings 
- Cosine Simularity 
- String formatting 
- Tidying up 
Estimated completion - by end of week

File naming convention
Can be found in gendoc.py

Results and discussion
Vocabulary restriction.
For vocabulary restriction I have removed stop words, these are words that are very common such as 'the' and 'a'. These words do not really give any meaning and will be very common which could give us false information. 

The results are PRELIMINARY, I am not very confident on the results and will take more time to asses
Results tables

|      Grain to Grain Similarity      | Topn  | Tdidf | Truncatedm=100 | Truncatedm=1000 | Similarity % |
|:-----------------------------------:|-------|-------|----------------|-----------------|--------------|
| vectorfileraw.csv                   | False | False | False          | False           | 35.81        |
| vectorfileraw_idf.csv               | False | True  | False          | False           | 10.50        |
| vectorfile_top20.csv                | True  | False | False          | False           | 50.94        |
| vectorfileraw_idf.csv               | False | True  | False          | False           | 10.50        |
| vectorfile_raw_tidf_top.csv         | True  | True  | False          | False           | 37.52        |
| vectorfile_truncated_100.csv        | False | False | True           | False           | 35.81        |
| vectorfiletruncated_m1000.csv       | False | False | False          | True            | 35.81        |
| vectorfiletruncatedm100_tdidf.csv   | False | True  | True           | False           | 10.50        |
| vectorfiletruncated_m1000_tdidf.csv | False | True  | False          | True            | 10.50        |

|      Crude to Crude Similarity      | Topn  | Tdidf | Truncatedm=100 | Truncatedm=1000 | Similarity % |
|:-----------------------------------:|-------|-------|----------------|-----------------|--------------|
| vectorfileraw.csv                   | False | False | False          | False           | 38.83        |
| vectorfileraw_idf.csv               | False | True  | False          | False           | 41.48        |
| vectorfile_top20.csv                | True  | False | False          | False           | 55.79        |
| vectorfileraw_idf.csv               | False | True  | False          | False           | 10.98        |
| vectorfile_raw_tidf_top.csv         | True  | True  | False          | False           | 37.52        |
| vectorfile_truncated_100.csv        | False | False | True           | False           | 38.83        |
| vectorfiletruncated_m1000.csv       | False | False | False          | True            | 38.83        |
| vectorfiletruncatedm100_tdidf.csv   | False | True  | True           | False           | 10.98        |
| vectorfiletruncated_m1000_tdidf.csv | False | True  | False          | True            | 10.98        |

|      Grain to Crude Similarity      | Topn  | Tdidf | Truncatedm=100 | Truncatedm=1000 | Similarity % |
|:-----------------------------------:|-------|-------|----------------|-----------------|--------------|
| vectorfileraw.csv                   | False | False | False          | False           | 33.11        |
| vectorfileraw_idf.csv               | False | True  | False          | False           | 7.66         |
| vectorfile_top20.csv                | True  | False | False          | False           | 47.90        |
| vectorfileraw_idf.csv               | False | True  | False          | False           | 10.98        |
| vectorfile_raw_tidf_top.csv         | True  | True  | False          | False           | 31.23        |
| vectorfile_truncated_100.csv        | False | False | True           | False           | 33.11        |
| vectorfiletruncated_m1000.csv       | False | False | False          | True            | 33.11        |
| vectorfiletruncatedm100_tdidf.csv   | False | True  | True           | False           | 7.66         |
| vectorfiletruncated_m1000_tdidf.csv | False | True  | False          | True            | 7.66         |

The hypothesis in your own words
*pending*

Bonus answers
*pending*
