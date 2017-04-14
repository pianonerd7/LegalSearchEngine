This is the README file for A0118888J-A0146123R-A0163945W's submission
Email: A0118888@u.nus.edu, E0010245@u.nus.edu, E0147987@u.nus.edu

== Python Version ==

We're using Python Version <3.6.0> for
this assignment.

== General Notes about this assignment ==

Some code is recycled from HW3 to prevent reinventing the wheel.

When indexing a document, we first parse the XML file and get values in
content, court and tags field. Then we will process the content by generating 
the term positional index table and calculate the document length of the content. 
We also index info in court field and whether the document contains tags.

In addition, considering the large number of documents in corpus, we will 
periodically write dictionaries and postings to disk, rather than keep them in memory
throughout indexing. After we index all files, we will merge dictionaries 
into one dictionary, postings into one posting and write them to the disk.

When retrieving a document, we apply such a strategy:
1. Perform basic tf-idf calculation with positional index. 
We get the document IDs that satisfy the positional constraints and remove the rest from the
score_dictionary. Also, because phrases are connected by AND operator, documents that don't
contain all query phrases will be removed.

2. Query expansion, by using synonyms of original query phrases/terms to perform tf-idf calculation.
In this round, we are not using positional index, and terms are connected by "OR" operator
(similar to HW 3). Also, we only calculate the tf-idf for those documents who match the query.
The tf-idf score obtained in this round will be multiplied by a small weight and 
added to original score.

3. Normalize score.

4. Add score if the distance between query phrases in the content of a document is small 
(within 20 words (a sentence) or within 100 words (a paragraph)).

5. Add score if the case is happened in a "more relevant court", i.e. in SGCA, SGHC, SG**, 
or CA in other country. Add score if the document contains tags, because as stated 
in the instruction for the assignment, documents with tagged are considered as 
landmark cases and are morelikly to be relevant.

Those parameters for score and weight are tested on the training data 
to achieve a better performance. However, because there's not enough of training data,
those parameters may not be optimized in a general situation.

Run-time optimizations:
We use xml.etree.ElementTree.iterparse and an iterator to parse XML, because it
is more memory efficient.

Allocation of work:
A0163945W: Indexing
A0118888J: tf-idf calculation with positional index and AND operator.
A0146123R: Query expansion, update score by position and field, clean-up and fix bugs.

== Files included with this submission ==

index.py 
- reads every file in the reuter folder, indexes, and write to disk.

node.py 
- represents every dictionary object.

queryParser.py 
- takes a query file and returns a list of list of words representing a 
list of query.

README.txt
- current file. Includes statement of individual work as well as general
notes on the assignment. 

search.py
- runs queries on dictionary-file and postings-file and write results into disk.

utility.py
- contain helper methods and constant storing.

dictionary.txt
- contains the dictionary constructed on data set.

postings.txt
- contains the postings constructed on data set.

== Statement of individual work ==

Please initial one of the following statements.

[X] I, A0118888J-A0146123R-A0163945W, certify that I have followed the CS 3245 Information
Retrieval class guidelines for homework assignments.  In particular, I
expressly vow that I have followed the Facebook rule in discussing
with others in doing the assignment and did not take notes (digital or
printed) from the discussions.  

[ ] I, A0000000X, did not follow the class rules regarding homework
assignment, because of the following reason:

<Please fill in>

I suggest that I should be graded as follows:

We completed all the work and adhere to the policy stated above.

== References ==

Lecture notes
https://docs.python.org/3/
http://stackoverflow.com/
