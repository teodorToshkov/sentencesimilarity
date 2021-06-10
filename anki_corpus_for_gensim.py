import os
import re
import numpy as np
import random

bg_stopwords = 'а автентичен аз ако ала бе без беше би бивш бивша бившо бил била били било благодаря близо бъдат бъде бяха в вас ваш ваша вероятно вече взема ви вие винаги внимава време все всеки всички всичко всяка във въпреки върху г ги главен главна главно глас го година години годишен д да дали два двама двамата две двете ден днес дни до добра добре добро добър докато докога дори досега доста друг друга други е евтин едва един една еднаква еднакви еднакъв едно екип ето живот за забавям зад заедно заради засега заспал затова защо защото и из или им има имат иска й каза как каква какво както какъв като кога когато което които кой който колко която къде където към лесен лесно ли лош м май малко ме между мек мен месец ми много мнозина мога могат може мокър моля момента му н на над назад най направи напред например нас не него нещо нея ни ние никой нито нищо но нов нова нови новина някои някой няколко няма обаче около освен особено от отгоре отново още пак по повече повечето под поне поради после почти прави пред преди през при пък първата първи първо пъти равен равна с са сам само се сега си син скоро след следващ сме смях според сред срещу сте съм със също т т.н. тази така такива такъв там твой те тези ти то това тогава този той толкова точно три трябва тук тъй тя тях у утре харесва хиляди ч часа че често чрез ще щом юмрук я як'.split()
fr_stopwords = 'au aux avec ce ces dans de des du elle en et eux il je la le leur lui ma mais me même mes moi mon ne nos notre nous on ou par pas pour qu que qui sa se ses son sur ta te tes toi ton tu un une vos votre vous c d j l à m n s t y été étée étées étés étant étante étants étantes suis es est sommes êtes sont serai seras sera serons serez seront serais serait serions seriez seraient étais était étions étiez étaient fus fut fûmes fûtes furent sois soit soyons soyez soient fusse fusses fût fussions fussiez fussent ayant ayante ayantes ayants eu eue eues eus ai as avons avez ont aurai auras aura aurons aurez auront aurais aurait aurions auriez auraient avais avait avions aviez avaient eut eûmes eûtes eurent aie aies ait ayons ayez aient eusse eusses eût eussions eussiez eussent'.split()
en_stopwords = 'i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she her hers herself it its itself they them their theirs themselves what which who whom this that these those am is are was were be been being have has had having do does did doing a an the and but if or because as until while of at by for with about against between into through during before after above below to from up down in out on off over under again further then once here there when where why how all any both each few more most other some such no nor not only own same so than too very s t can will just don should now d ll m o re ve y ain aren couldn didn doesn hadn hasn haven isn ma mightn mustn needn shan shouldn wasn weren won wouldn'.split()

class AnkiCorpus:

    def __init__(self, name, window_size=2):

        self.start = 0

        file_path = os.path.join(os.path.curdir, "Corpus", name)
        file = open(file_path, encoding='utf8')

        self.text = file.readlines()
        self.raw_text = self.text

        self.format_text()

        self.text_to_data()

        random.shuffle(self.data)

    def format_text(self):
        for i in range(len(self.text)):
            self.text[i] = self.text[i].lower()
            self.text[i] = re.sub(r'[\n]+', ' ', self.text[i])
            self.text[i] = re.sub(r'[\_\\\/\…\— \-\,\*\„\«\»]+', ' ', self.text[i])
            self.text[i] = re.sub(r' *[\(\)\[\]\{\}\"\“\”]+ *', ' ', self.text[i])
            self.text[i] = re.sub(r'[\'\‘\’]', '\'', self.text[i])
            self.text[i] = re.sub(r'\.+\'+', ' ', self.text[i])
            self.text[i] = re.sub(r' +\'+', ' ', self.text[i])
            self.text[i] = re.sub(r'\'+\.+', ' ', self.text[i])
            self.text[i] = re.sub(r'\'+ +', ' ', self.text[i])
            self.text[i] = re.sub(r'[\?\;\!\.]+', ' ', self.text[i])
            self.text[i] = re.sub(r'\:+', '', self.text[i])
            self.text[i] = re.sub(r' *\.+ *', ' ', self.text[i])
            self.text[i] = re.sub(r' +', ' ', self.text[i])

    def text_to_data(self):

        raw_data = [x.split('\t') for x in self.text]

        self.data = []

        for sentences in raw_data:
            sentences = [x.split() for x in sentences]
            self.data.append(sentences)

    def get_data(self, en=True):
        if en == True:
            i = 0
        else:
            i = 1
        return [x[i] for x in self.data]

    def sent2vec(self, sentence, gensim_model, stopwords):
        vec = np.zeros(gensim_model.vector_size)
        for word in sentence:
            if word in gensim_model.vocab\
                    and word not in stopwords:
                vec += gensim_model[word]
        return vec

    def euclidean_dist(self, vec1, vec2):
        return np.sqrt(np.sum((vec1 - vec2) ** 2))

    def find_n_neighbours(self, word_vec, gensim_model, n=10):
        min_dist = 10000  # to act like positive infinity
        min_index = -1

        close_words = []

        words = [val for val in gensim_model.wv.vocab]

        for index, vector in enumerate(gensim_model[gensim_model.wv.vocab]):
            if self.euclidean_dist(vector, word_vec) < min_dist and not np.array_equal(vector, word_vec):
                # min_dist = euclidean_dist(vector, query_vector)
                # min_index = index
                close_words.append([words[index], self.euclidean_dist(vector, word_vec)])
        # return [id2word(min_index), min_dist]
        close_words.sort(key= lambda l:l[1])
        return  close_words[:n]

    def next_batch(self, length, en_model, other_model):
        x = []
        y = []

        if self.start >= len(self.data):
            raise IndexError('The start index is ' +\
                             str(self.start) +\
                             ' but the length of the data is ' +\
                             str(len(self.data)))
        else:
            end = self.start + length
            for d in self.data[self.start : min(end, len(self.data))]:
                x.append(self.sent2vec(d[1], other_model))
                y.append(self.sent2vec(d[0], en_model))
            while end >= len(self.data):
                end -= len(self.data)
                for d in self.data[0 : min(end, len(self.data))]:
                    x.append(self.sent2vec(d[1], other_model))
                    y.append(self.sent2vec(d[0], en_model))

        if len(x) != length:
            raise BufferError('The length required is ' +\
                              str(length) +\
                              " but the actual length is " +\
                              str(len(x)))

        x = np.asarray(x)
        y = np.asarray(y)

        self.start = end

        return [x, y]
