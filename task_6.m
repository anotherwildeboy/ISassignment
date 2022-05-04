
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
