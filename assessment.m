clc
clear
%PART 1 DATA INPUT
fprintf("DATA INPUT \n");
% Read positive words
fidPositive = fopen(fullfile('opinion-lexicon-English','positive-words.txt'));
C = textscan(fidPositive,'%s','CommentStyle',';');
wordsPositive = string(C{1});

% Read negative words
fidNegative = fopen(fullfile('opinion-lexicon-English','negative-words.txt'));
C = textscan(fidNegative,'%s','CommentStyle',';');
wordsNegative = string(C{1});
fclose all;

%PART 2 DATA PROCESSING
fprintf("DATA PREPROCESSING \n");
filename = "yelp_labelled.txt";
dataReviews = readtable(filename,'TextType','string');
textData = dataReviews.review; %get review text
actualScore = dataReviews.score; %get human reviewer's sentiment score

% Convert the text data to lowercase.
cleanTextData = lower(textData);

% Tokenize the text.
documents = tokenizedDocument(cleanTextData);

% Erase punctuation.
documents = erasePunctuation(documents);

% Remove a list of stop words.

documents = removeStopWords(documents);

sents = documents;

%PART 3 WORD EMBEDDING SETUP
fprintf("WORD EMBEDDING \n");
rng('default')
emb = fastTextWordEmbedding;

% Create table of labeled words
words = [wordsPositive;wordsNegative];
labels = categorical(nan(numel(words),1));
labels(1:numel(wordsPositive)) = "Positive";
labels(numel(wordsPositive)+1:end) = "Negative";

data = table(words,labels,'VariableNames',{'Word','Label'});
idx = ~isVocabularyWord(emb,data.Word); % Matlab 18b
data(idx,:) = [];

%PART 4 TRAINING SVM
fprintf("TRAINING \n");
%Set aside 10% of the words at random for testing.
numWords = size(data,1);
cvp = cvpartition(numWords,'HoldOut',0.3); %holdout fewer if applying model CHANGE VALUE HERE TO CHANGE TRAINING:TEST RATIO
dataTrain = data(training(cvp),:);
dataTest = data(test(cvp),:);

%Convert the words in the training data to word vectors using word2vec.
wordsTrain = dataTrain.Word;
XTrain = word2vec(emb,wordsTrain);
YTrain = dataTrain.Label;

%Train a support vector machine (SVM) Sentiment Classifier which classifies word vectors into positive and negative categories.
model = fitcsvm(XTrain,YTrain);

%PART 5 TESTING SVM
fprintf("TESTING \n");
idx = ~isVocabularyWord(emb,sents.Vocabulary); %18b
removeWords(sents, idx);

sentimentScore = zeros(size(sents));

for ii = 1 : sents.length
    docwords = sents(ii).Vocabulary;
    vec = word2vec(emb,docwords);
    [~,scores] = predict(model,vec);
    sentimentScore(ii) = mean(scores(:,1));
    if isnan(sentimentScore(ii))
        sentimentScore(ii) = 0;
    end
    fprintf('Sent: %d, words: %s, FoundScore: %d, GoldScore: %d\n', ii, joinWords(sents(ii)), sentimentScore(ii), actualScore(ii));
end

%PART 6 RESULTS DISPLAY
fprintf("RESULTS \n");
%coerce sentiment to 1/0 scale

sentimentScore(sentimentScore > 0) = 1;   %take >1 to be 1 only
sentimentScore(sentimentScore < 0)= -1;   %there is no neutral only negative

notfound = sum(sentimentScore == 0);
covered = numel(sentimentScore) - notfound;

%coerce data to match scales
tp = sentimentScore((sentimentScore > 0) & ( actualScore >0));
tn = sentimentScore((sentimentScore  < 0) &( actualScore == 0));

%coverage

fprintf("Coverage: %2.2f%%, found  %d, missed: %d\n", (covered * 100)/numel(sentimentScore), covered, notfound);

%Calculate accuracy
acc = (sum(tp) - sum(tn))/sum(covered);
fprintf("Accuracy: %2.2f%%, tp: %d, tn: %d\n", acc*100, sum(tp), -sum(tn));


figure
confusionchart(actualScore, sentimentScore);