#

<h1>LT2212 V19 Assignment 2</h1>
<h3>From Asad Sayeed's statistical NLP course at the University of Gothenburg.</h3>

<h2>Name: Robert Rhys Thomas</h2> 

<h4>Additional instructions:</h4>
<p>There are no additional instruction. Program runs as specified however the option to name the file has been removed
as I have created filenames depending on what function has been operated on the file. For example if -B20 are accepted the file will be named 'vectorfile_top20' as the top20 words function will be applied</p>

<h4>File naming convention:</h4>
All filenames are easily legible in the zip file however tf-idf might be referred to as td-idf, rest assured they are the same I just managed to put it in wrong for 3/4 of the programming and it would have been quite a challenge to correct it.

<h4>Results:</h4>


|                               | grain to grain | crude to crude | crude to grain | grain to crude |
|-------------------------------|----------------|----------------|----------------|----------------|
| raw counts                    | 0.395          | 0.457          | 0.331          | 0.331          |
| top20 counts                  | 0.704          | 0.79           | 0.622          | 0.622          |
| tfidf counts                  | 0.093          | 0.112          | 0.077          | 0.077          |
| top20 tfidf counts            | 0.664          | 0.755          | 0.55           | 0.55           |
| truncated raw counts n=100    | 0.551          | 0.615          | 0.438          | 0.438          |
| truncated tfidf counts n=1000 | 0.258          | 0.32           | 0.183          | 0.183          |
| truncated counts n=1000       | 0.385          | 0.453          | 0.321          | 0.321          |
| truncated tfidf counts n=1000 | 0.091          | 0.111          | 0.074          | 0.074          |

<h4>Discussion:</h4>
<p>The results, if calculations are correct, provide some interesting trends when applying different methods of feature extraction. I am not that confident in the output due to my own hypothesis, for example I would expect the files, when truncated and tfidf applied, to resemble their own topic with an accuracy highter than 0.091, however there are a few results that do conform to my expectations. When top20 is applied to the raw counts and the tfidf counts the documents resemble the own topics to an accuracy of above 0.6, I find this reasonable as the program takes a much smaller data set into account therefore filtering out tokens that might affect results negatively. Truncation returned some intersting results, truncation resulted in the files bein a lot less similar to themselves and also a lot less similar between topics, this is possibly due to a bug or perhaps it is because the truncation is the data that makes the files similar. 
  
<h4>Vocabulary restriction:</h4>
<p>For vocabulary restriction I have removed stop words, these are words that are very common such as 'the' and 'a'. These words do not really give any meaning and will be very common which could give us false information. I have also removed numbers as I found, through scanning through the documents, that there was a lot of numbers that didn't seem to give any unique insite into the text file and they also took up quite a few places in the top words so I thought it better to remove them entirerly. </p>

<h4>Hypothesis:</h4>
<p>This exercise has been an experiment to see what methods there are available to pass data from input to output and the effects that certain methods will have. We have learnt that we can not analys text files without preprocessing them, data can be lost when words with particular meaning or function are given the same weight as perhaps words that give more enlightening information about an input. For example in this assignment by removing stopwords, punctuation and applying tf-df we are given much relevant and reliable results.</p>

<h5>Bonus answers</h5>
<p>*pending*<p>

<h4>Discussion of trends in results in light of the hypothesis:</h4>
<p>The results certainly suggest that by using different statistical methods the result can vary somewhat. If we assume my calculations are correct we can see that the more functions that are used the greater the difference between the orginal raw count results and the statisticaly manipulated result. We can see that the more functions added the topics become less similar to themselves and less similar to the other topic. This is a rather strange result which either suggests there is a bug that I have missed or that the program is stripping away all the features that make the vectors recognisable as either crude or grain. </p>

