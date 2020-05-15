Name	: Chaitanya Kulkarni
B-Number: B00814455
Email	: ckulkar2@binghamton.edu
------------------------------------------------------------------------------------
Academic Honesty Statement:
I have done this assignment completely on my own. I have not copied it, nor have I given my solution to anyone else.
I understand that if I am involved in plagiarism or cheating I will have to sign an official form that
I have cheated and that this form will be stored in my official university record.
I also understand that I will receive a grade of 0 for the involved assignment for my first offense and that
I will receive a grade of “F” for the course for any additional offense.
-Chaitanya Kulkarni
------------------------------------------------------------------------------------
To Use dataset 1, set following path -
    - decision.py (Line 56)
        PATH = "dataset1/"
To Use dataset 2, set following path -
    - decision.py (Line 56)
        PATH = "dataset2/"
------------------------------------------------------------------------------------
How to Compile and run the program
- program is written in python 3.7
- program is tested on remote.cs.binghamton.edu and is running successfully as expected

in command prompt type-
    >> python3 decisionTree.py <training-set> <test-set> <to-print>:{yes,no} heuristic
    >> python3 decision.py training_set.csv test_set.csv yes entropy

    To display Information Gain Accuracy along with Tree, type -
        >> python3 decision.py training_set.csv test_set.csv yes entropy
    To display Information Gain Accuracy without Tree, type -
        >> python3 decision.py training_set.csv test_set.csv no entropy
    To display Variance Impurity Accuracy along with Tree, type -
        >> python3 decision.py training_set.csv test_set.csv yes variance
    To display Variance Impurity Accuracy without Tree, type -
        >> python3 decision.py training_set.csv test_set.csv no variance
--------------------------------------------------------------------------------------
Output
- Program take around 2-3 minutes to execute completely.
- Information Gain Heuristic is displayed on console if input is "entropy"
- Variance Impurity Accuracy is displayed on console if input is "variance"
- Both IG and Variance input is stored in the Result.txt text file.