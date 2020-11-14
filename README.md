# Translator_CH-EN   
This is a Chinese-English Translator demo, refferred to the thesis:[Attention Is All You Need](https://arxiv.org/abs/1706.03762).    
I tried to build rebuild a model as the thesis mentioned, and make a simple user interface for the application, the UI like this:   

<img src="pic/UI.png" width = 650/>  

The system run as following system flow chart:   

<img src="pic/system_flow_chart.png" width = 650/>  

For chinese Translator, we need to add word_segmentation function to unified sentence structure, as following picture:   

<img src="pic/word_segmentation.png" width = 650/>  

Then, we need preprocess as other kind of Translator(Count the number of occurrences of each word):   

<img src="pic/result_preproccess.png" width = 650/>  

After that, we can train the model:   

<img src="pic/training.png" width = 650/>  

The total result:   

<img src="pic/total_result.png" width = 650/>  

We could get some translation as following(similar meaning in different result, but ignore draft):   

<img src="pic/diff_result_same_meaning.png" width = 650/>  

Good result for some sentences that have appeared or stereotyped expression:   

<img src="pic/good_result_stereotyped_expression.png" width = 650/>  

However, the application had some problems with the punctuation, time and number:   

<img src="pic/problem_end_sentence.png" width = 650/>  

<img src="pic/probelm_time.png" width = 650/>  

This problem can be solved if you add a function to find all punctuations, times and numbers in preprocess function. If you want to have better result, please add that yourself.
