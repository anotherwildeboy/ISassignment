% Read positive words
fidPositive = fopen(fullfile('opinion-lexicon-English','positive-words.txt'));
C = textscan(fidPositive,'%s','CommentStyle',';');
wordsPositive = string(C{1});

% Read negative words
fidNegative = fopen(fullfile('opinion-lexicon-English','negative-words.txt'));
C = textscan(fidNegative,'%s','CommentStyle',';');
wordsNegative = string(C{1});
fclose all;

%https://undocumentedmatlab.com/blog/using-java-collections-in-matlab

words_hash = java.util.Hashtable;
[possize, ~] = size(wordsPositive);
[negsize,~] = size(wordsNegative);
for ii = 1:possize
    words_hash.put(wordsPositive(ii,1),1);
end
for ii = 1:negsize
    words_hash.put(wordsNegative(ii,1),-1);
end
