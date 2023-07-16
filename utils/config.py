# Kmeans
n_clusters = 140

# Word2Vec
min_count = 10 # mínimo de ocorrências para a palavra participar
window = 7 # janela de vetorização 
n_dim = 50 # dimensões do espaço vetorial

# Limites de palavras para um comentário tratado ser classificado
lower_limit = 3
upper_limit = 50


# Substituições extras (regex) para evitar clusters desnecessários (depois da lematização e do Unidecode). Adicionar sinônimos e typos.
dictionary = {}

remove_from_stopword_list = [] # são stopwords, mas contribuem para o sentido do texto, portanto não são excluídas

add_to_stopword_list = []

useless_terms = ['(^|[^A-Za-zÀ-ÿ])(ola)[^A-Za-zÀ-ÿ]', # preceded by start of line or something that is not a letter
    '(^|[^A-Za-zÀ-ÿ])(ano)[^A-Za-zÀ-ÿ]', 
    '(^|[^A-Za-zÀ-ÿ])(amar)[^A-Za-zÀ-ÿ]'
] # remoção por regex

blacklist = ['html', '.com', '.us', 'utm', 'utmsourc', 'http'] # frases contendo esses termos são ignoradas (executada após unidecode)

# palavras obrigatórias para classificação (lematizadas e sem acento)
mandatory = []

ignorar = [] # ignorar na lematização


# Criação manual dos "clusters", que são compostos pelos resultados do K-means
'''
Não importa a ordem que os termos stemizados são colocados. O algoritmo dá prioridade aos termos mais frequentes dos subclusters criados pelo K-means.

Formato dict('key': list)
'''

clusters = {
    'Climate Change': ['climat chang', 'sea level'],
    'Believing anything': ['believ anything'],
    'Education': ['high school'],
    'Government': ['foreign government', 'feder government'],
    'Fake News': ['fak new', 'lie lie'],
    'Money': ['money money'],
    'United States': ['united stat', 'american peopl'],
    'Compliments': ['good work'],
    'Clinton': ['bill clinton', 'hillary clinton'],
    'Right wing': ['right wing'],
    'Middle East': ['middl east'],
    'Left wing': ['left wing'],
    'Swearing': ['piec shit', 'douch bag', 'stupid stupid'],
    'Donald Trump': ['donald trump', 'trump family', 'president trump', 'trump jr', 'trump said'],
    'Death penalty': ['death penalty'],
    'Child abuse': ['child abus'],
    'Terrorism': ['terrorist attack'],
    'White House': ['whit hous'],
    'Taxes': ['tax cut', 'taxpay money', 'tax pay'],
    'Religion': ['jesu christ'],
    'Lies': ['lie lie'],
    'Barack Obama': ['barack obam', 'obam administration', 'president obam'],
    'Justice/Law enforcement': ['law enforcement', 'suprem court'],
    'North Korea': ['north kore'],
    'Cyber Security': ['cyb security'],
    'Political Parties': ['republican party', 'democratic party'],
    'Mental health': ['ment illnes'],
    'Money': ['million doll', 'billion dol'],
    'Crime': ['obstruction just', 'go jail'],
    'Sex assault': ['sex assault'],
    'White people': ['whit peopl'],
    'Senate/Congress': ['hous senat', 'republican congres'],
    'Election': ['vot fraud'],
    'Russia': ['russian government', 'russ russ', 'putin trump'],
    'US Border Closing': ['build wall'],
    'Social Politics': ['minimum wag', 'health car'],
    'Immigration': ['illeg immigrant'],
    'Paul Ryan': ['paul ryan'],
    'France': ['mr macron'],
    'Freedom of speech': ['fre speech', 'fre world'],
    'War': ['civil war', 'cold war'],
    'Diet': ['eat meat'],
    'News/TV': ['fox new', 'tv show']

}