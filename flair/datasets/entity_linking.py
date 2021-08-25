import csv
import logging
import os
from pathlib import Path
from typing import Union

import requests

import flair
from flair.data import Dictionary, Sentence
from flair.datasets import ColumnCorpus
from flair.file_utils import cached_path, unpack_file
from flair.tokenization import SentenceSplitter, SegtokSentenceSplitter

log = logging.getLogger("flair")


class EntityLinkingCorpus(ColumnCorpus):
    def __init__(
            self,
            data_folder,
            train_file,
            columns={0: "text", 1: "nel"},
            column_delimiter="\t",
            in_memory=True,
            document_separator_token='-DOCSTART-',
            **corpusargs,
    ):
        """
        Super class for all entity linking corpora. Expects the data to be in column format with one column for words and another one for BIO-tags and wikipedia-page
        name, e.g. B-Brad_Pitt.
        The class provides the function make_entity_dict to create an entity dictionary suited for entity linking.
        """
        # TODO: Add a routine, that checks annotations for some widespread errors/inconsistencies??? (e.g. in AQUAINT corpus Iran-Iraq_War vs. Iran-Iraq_war)

        super(EntityLinkingCorpus, self).__init__(
            data_folder,
            columns,
            train_file=train_file,
            column_delimiter=column_delimiter,
            in_memory=in_memory,
            document_separator_token=document_separator_token,
            **corpusargs,
        )

    def make_entity_dict(self, threshold: int = 1, mode=False) -> Dictionary:
        """
        Create ID-dictionary for the wikipedia-page names.
        param threshold: Ignore links that occur less than threshold value

        In entity_occurences all wikinames and their number of occurence is saved.
        ent_dictionary contains all wikinames that occure at least threshold times and gives each name an ID
        """
        self.threshold = threshold
        self.entity_occurences = {}
        self.total_number_of_entity_mentions = 0

        for sentence in self.get_all_sentences():
            if not sentence.is_document_boundary:  # exclude "-DOCSTART-"-sentences

                spans = sentence.get_spans('nel')
                for span in spans:
                    annotation = span.tag
                    self.total_number_of_entity_mentions += 1
                    if annotation in self.entity_occurences:
                        self.entity_occurences[annotation] += 1
                    else:
                        self.entity_occurences[annotation] = 1

        self.number_of_entities = len(self.entity_occurences)

        # Create the annotation dictionary
        self.ent_dictionary: Dictionary = Dictionary(add_unk=True)

        for x in self.entity_occurences:
            if self.entity_occurences[x] >= threshold:
                self.ent_dictionary.add_item(x)

        return self.ent_dictionary

    # this fct removes every second unknown label
    def remove_unknowns(self):
        remove = True
        for sentence in self.get_all_sentences():
            if not sentence.is_document_boundary:  # exclude "-DOCSTART-"-sentences

                spans = sentence.get_spans('nel')
                for span in spans:
                    annotation = span.tag
                    if self.ent_dictionary.get_idx_for_item(annotation) == 0:  # unknown label
                        if remove:
                            for token in span:
                                token.remove_labels('nel')
                            remove = False
                        else:
                            remove = True


class NEL_ENGLISH_AQUAINT(EntityLinkingCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            in_memory: bool = True,
            agreement_threshold: float = 0.5,
            sentence_splitter: SentenceSplitter = SegtokSentenceSplitter(),
            **corpusargs,
    ):
        """
        Initialize Aquaint Entity Linking corpus introduced in: D. Milne and I. H. Witten.
        Learning to link with wikipedia
        (https://www.cms.waikato.ac.nz/~ihw/papers/08-DNM-IHW-LearningToLinkWithWikipedia.pdf).
        If you call the constructor the first time the dataset gets automatically downloaded and transformed in
        tab-separated column format (aquaint.txt).

        Parameters
        ----------
        base_path : Union[str, Path], optional
            Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
            to point to a different folder but typically this should not be necessary.
        in_memory: If True, keeps dataset in memory giving speedups in training.
        agreement_threshold: Some link annotations come with an agreement_score representing the agreement from the human annotators. The score ranges from lowest 0.2
                             to highest 1.0. The lower the score, the less "important" is the entity because fewer annotators thought it was worth linking.
                             Default is 0.5 which means the majority of annotators must have annoteted the respective entity mention.
        """
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        self.agreement_threshold = agreement_threshold

        # this dataset name
        dataset_name = self.__class__.__name__.lower() + "_" + type(sentence_splitter).__name__

        # default dataset folder is the cache root
        if not base_path:
            base_path = flair.cache_root / "datasets"
        data_folder = base_path / dataset_name

        aquaint_el_path = "https://www.nzdl.org/wikification/data/wikifiedStories.zip"
        corpus_file_name = "aquaint.txt"
        parsed_dataset = data_folder / corpus_file_name

        # download and parse data if necessary
        if not parsed_dataset.exists():
            aquaint_el_zip = cached_path(f"{aquaint_el_path}", Path("datasets") / dataset_name)
            unpack_file(aquaint_el_zip, data_folder, "zip", False)

            try:
                with open(parsed_dataset, "w", encoding='utf-8') as txt_out:

                    # iterate over all html files
                    for file in os.listdir(data_folder):

                        if not file.endswith(".htm"):
                            continue

                        with open(str(data_folder / file), "r", encoding='utf-8') as txt_in:
                            text = txt_in.read()

                        # get rid of html syntax, we only need the text
                        strings = text.split("<p> ")
                        strings[0] = strings[0].split('<h1 id="header">')[1][:-7]

                        for i in range(1, len(strings) - 1):
                            strings[i] = strings[i][:-7]

                        strings[-1] = strings[-1][:-23]

                        # between all documents we write a separator symbol
                        txt_out.write('-DOCSTART-\n\n')

                        for string in strings:

                            # skip empty strings
                            if not string: continue

                            # process the annotation format in the text and collect triples (begin_mention, length_mention, wikiname)
                            indices = []
                            lengths = []
                            wikinames = []

                            current_entity = string.find('[[')  # each annotation starts with '[['
                            while current_entity != -1:
                                wikiname = ''
                                surface_form = ''
                                j = current_entity + 2

                                while string[j] not in [']', '|']:
                                    wikiname += string[j]
                                    j += 1

                                if string[j] == ']':  # entity mention ends, i.e. looks like this [[wikiname]]
                                    surface_form = wikiname  # in this case entity mention = wiki-page name
                                else:  # string[j] == '|'
                                    j += 1
                                    while string[j] not in [']', '|']:
                                        surface_form += string[j]
                                        j += 1

                                    if string[
                                        j] == '|':  # entity has a score, i.e. looks like this [[wikiname|surface_form|agreement_score]]
                                        agreement_score = float(string[j + 1:j + 4])
                                        j += 4  # points to first ']' of entity now
                                        if agreement_score < self.agreement_threshold:  # discard entity
                                            string = string[:current_entity] + surface_form + string[j + 2:]
                                            current_entity = string.find('[[')
                                            continue

                                # replace [[wikiname|surface_form|score]] by surface_form and save index, length and wikiname of mention
                                indices.append(current_entity)
                                lengths.append(len(surface_form))
                                wikinames.append(wikiname[0].upper() + wikiname.replace(' ', '_')[1:])

                                string = string[:current_entity] + surface_form + string[j + 2:]

                                current_entity = string.find('[[')

                            # sentence splitting and tokenization
                            sentences = sentence_splitter.split(string)
                            sentence_offsets = [sentence.start_pos for sentence in sentences]

                            # iterate through all annotations and add to corresponding tokens
                            for mention_start, mention_length, wikiname in zip(indices, lengths, wikinames):

                                # find sentence to which annotation belongs
                                sentence_index = 0
                                for i in range(1, len(sentences)):
                                    if mention_start < sentence_offsets[i]:
                                        break
                                    else:
                                        sentence_index += 1

                                # position within corresponding sentence
                                mention_start -= sentence_offsets[sentence_index]
                                mention_end = mention_start + mention_length

                                # set annotation for tokens of entity mention
                                first = True
                                for token in sentences[sentence_index].tokens:
                                    if token.start_pos >= mention_start and token.end_pos <= mention_end:  # token belongs to entity mention
                                        if first:
                                            token.set_label(typename='nel', value='B-' + wikiname)
                                            first = False
                                        else:
                                            token.set_label(typename='nel', value='I-' + wikiname)

                            # write to out-file in column format
                            for sentence in sentences:

                                for token in sentence.tokens:

                                    labels = token.get_labels('nel')

                                    if len(labels) == 0:  # no entity
                                        txt_out.write(token.text + '\tO\n')

                                    else:  # annotation
                                        txt_out.write(token.text + '\t' + labels[0].value + '\n')

                                txt_out.write('\n')  # empty line after each sentence

            except:
                # in case something goes wrong, delete the dataset and raise error
                os.remove(parsed_dataset)
                raise

        super(NEL_ENGLISH_AQUAINT, self).__init__(
            data_folder,
            train_file=corpus_file_name,
            in_memory=in_memory,
            **corpusargs,
        )


class NEL_GERMAN_HIPE(EntityLinkingCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            in_memory: bool = True,
            wiki_language: str = 'dewiki',
            **corpusargs
    ):
        """
        Initialize a sentence-segmented version of the HIPE entity linking corpus for historical German (see description
        of HIPE at https://impresso.github.io/CLEF-HIPE-2020/). This version was segmented by @stefan-it and is hosted
        at https://github.com/stefan-it/clef-hipe.
        If you call the constructor the first time the dataset gets automatically downloaded and transformed in
        tab-separated column format.

        Parameters
        ----------
        base_path : Union[str, Path], optional
            Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
            to point to a different folder but typically this should not be necessary.
        in_memory: If True, keeps dataset in memory giving speedups in training.
        wiki_language : specify the language of the names of the wikipedia pages, i.e. which language version of
        Wikipedia URLs to use. Since the text is in german the default language is German.
        """
        self.wiki_language = wiki_language
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = flair.cache_root / "datasets"
        data_folder = base_path / dataset_name

        dev_raw_url = "https://raw.githubusercontent.com/stefan-it/clef-hipe/main/data/future/dev-v1.2/de/HIPE-data-v1.2-dev-de-normalized-manual-eos.tsv"
        test_raw_url = "https://raw.githubusercontent.com/stefan-it/clef-hipe/main/data/future/test-v1.3/de/HIPE-data-v1.3-test-de-normalized-manual-eos.tsv"
        train_raw_url = "https://raw.githubusercontent.com/stefan-it/clef-hipe/main/data/future/training-v1.2/de/HIPE-data-v1.2-train-de-normalized-manual-eos.tsv"
        train_file_name = wiki_language + "_train.tsv"
        parsed_dataset = data_folder / train_file_name

        # download and parse data if necessary
        if not parsed_dataset.exists():

            # from qwikidata.linked_data_interface import get_entity_dict_from_api

            original_train_path = cached_path(f"{train_raw_url}", Path("datasets") / dataset_name)
            original_test_path = cached_path(f"{test_raw_url}", Path("datasets") / dataset_name)
            original_dev_path = cached_path(f"{dev_raw_url}", Path("datasets") / dataset_name)

            # generate qid wikiname dictionaries
            log.info('Get wikinames from wikidata...')
            train_dict = self._get_qid_wikiname_dict(path=original_train_path)
            test_dict = self._get_qid_wikiname_dict(original_test_path)
            dev_dict = self._get_qid_wikiname_dict(original_dev_path)
            log.info('...done!')

            # merge dictionaries
            qid_wikiname_dict = {**train_dict, **test_dict, **dev_dict}

            for doc_path, file_name in zip([original_train_path, original_test_path, original_dev_path],
                                           [train_file_name, wiki_language + '_test.tsv', wiki_language + '_dev.tsv']):
                with open(doc_path, 'r', encoding='utf-8') as read, open(data_folder / file_name, 'w',
                                                                         encoding='utf-8') as write:

                    # ignore first line
                    read.readline()
                    line = read.readline()
                    last_eos = True

                    while line:
                        # commented and empty lines
                        if line[0] == '#' or line == '\n':
                            if line[2:13] == 'document_id':  # beginning of new document

                                if last_eos:
                                    write.write('-DOCSTART-\n\n')
                                    last_eos = False
                                else:
                                    write.write('\n-DOCSTART-\n\n')

                        else:
                            line_list = line.split('\t')
                            if not line_list[7] in ['_', 'NIL']:  # line has wikidata link

                                wikiname = qid_wikiname_dict[line_list[7]]

                                if wikiname != 'O':
                                    annotation = line_list[1][:2] + wikiname
                                else:  # no entry in chosen language
                                    annotation = 'O'

                            else:

                                annotation = 'O'

                            write.write(line_list[0] + '\t' + annotation + '\n')

                            if line_list[-1][-4:-1] == 'EOS':  # end of sentence
                                write.write('\n')
                                last_eos = True
                            else:
                                last_eos = False

                        line = read.readline()

        super(NEL_GERMAN_HIPE, self).__init__(
            data_folder,
            train_file=train_file_name,
            dev_file=wiki_language + '_dev.tsv',
            test_file=wiki_language + '_test.tsv',
            in_memory=in_memory,
            **corpusargs,
        )

    def _get_qid_wikiname_dict(self, path):

        qid_set = set()
        with open(path, mode='r', encoding='utf-8') as read:
            # read all Q-IDs

            # ignore first line
            read.readline()
            line = read.readline()

            while line:

                if not (line[0] == '#' or line == '\n'):  # commented or empty lines
                    line_list = line.split('\t')
                    if not line_list[7] in ['_', 'NIL']:  # line has wikidata link

                        qid_set.add(line_list[7])

                line = read.readline()

        base_url = 'https://www.wikidata.org/w/api.php?action=wbgetentities&format=json&props=sitelinks&sitefilter=' + self.wiki_language + '&ids='

        qid_list = list(qid_set)
        ids = ''
        length = len(qid_list)
        qid_wikiname_dict = {}
        for i in range(length):
            if (
                    i + 1) % 50 == 0 or i == length - 1:  # there is a limit to the number of ids in one request in the wikidata api

                ids += qid_list[i]
                # request
                response_json = requests.get(base_url + ids).json()

                for qid in response_json['entities']:

                    try:
                        wikiname = response_json['entities'][qid]['sitelinks'][self.wiki_language]['title'].replace(' ',
                                                                                                                    '_')
                    except KeyError:  # language not available for specific wikiitem
                        wikiname = 'O'

                    qid_wikiname_dict[qid] = wikiname

                ids = ''

            else:
                ids += qid_list[i]
                ids += '|'

        return qid_wikiname_dict


class NEL_ENGLISH_AIDA(EntityLinkingCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            in_memory: bool = True,
            check_existence: bool = False,
            **corpusargs
    ):
        """
        Initialize AIDA CoNLL-YAGO Entity Linking corpus introduced here https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/ambiverse-nlu/aida/downloads.
        License: https://creativecommons.org/licenses/by-sa/3.0/deed.en_US
        If you call the constructor the first time the dataset gets automatically downloaded and transformed in tab-separated column format.

        Parameters
        ----------
        base_path : Union[str, Path], optional
            Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
            to point to a different folder but typically this should not be necessary.
        in_memory: If True, keeps dataset in memory giving speedups in training.
        check_existence: If True the existence of the given wikipedia ids/pagenames is checked and non existent ids/names will be igrnored.
        """
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = flair.cache_root / "datasets"
        data_folder = base_path / dataset_name

        conll_yago_path = "https://nlp.informatik.hu-berlin.de/resources/datasets/conll_entity_linking/"
        corpus_file_name = "train"
        parsed_dataset = data_folder / corpus_file_name

        if not parsed_dataset.exists():

            import wikipediaapi

            wiki_wiki = wikipediaapi.Wikipedia(language='en')

            testa_unprocessed_path = cached_path(f"{conll_yago_path}aida_conll_testa", Path("datasets") / dataset_name)
            testb_unprocessed_path = cached_path(f"{conll_yago_path}aida_conll_testb", Path("datasets") / dataset_name)
            train_unprocessed_path = cached_path(f"{conll_yago_path}aida_conll_train", Path("datasets") / dataset_name)

            # we use the wikiids in the data instead of directly utilizing the wikipedia urls.
            # like this we can quickly check if the corresponding page exists
            wikiid_wikiname_dict = self._get_wikiid_wikiname_dict(data_folder)

            for name, path in zip(['train', 'testa', 'testb'],
                                  [train_unprocessed_path, testa_unprocessed_path, testb_unprocessed_path]):
                with open(data_folder / name, 'w', encoding='utf-8') as write, open(path, 'r',
                                                                                    encoding='utf-8') as read:

                    for line in read:

                        line_list = line.split('\t')
                        if len(line_list) <= 4:
                            if line_list[0][:10] == '-DOCSTART-':  # Docstart
                                write.write('-DOCSTART-\n\n')
                            elif line_list[0] == '\n':  # empty line
                                write.write('\n')
                            else:  # text without annotation or marked '--NME--' (no matching entity)
                                if len(line_list) == 1:
                                    write.write(line_list[0][:-1] + '\tO\n')
                                else:
                                    write.write(line_list[0] + '\tO\n')
                        else:  # line with annotation
                            wikiname = wikiid_wikiname_dict[line_list[5].strip()]
                            if wikiname != 'O':
                                write.write(line_list[0] + '\t' + line_list[1] + '-' + wikiname + '\n')
                            else:
                                # if there is a bad wikiid we can check if the given url in the data exists using wikipediaapi
                                wikiname = line_list[4].split('/')[-1]
                                if check_existence:
                                    page = wiki_wiki.page(wikiname)
                                    if page.exists():
                                        write.write(line_list[0] + '\t' + line_list[1] + '-' + wikiname + '\n')
                                    else:  # neither the wikiid nor the url exist
                                        write.write(line_list[0] + '\tO\n')
                                else:
                                    write.write(line_list[0] + '\t' + line_list[4] + '-' + wikiname + '\n')

                # delete unprocessed file
                os.remove(path)

        super(NEL_ENGLISH_AIDA, self).__init__(
            data_folder,
            train_file=corpus_file_name,
            dev_file='testa',
            test_file='testb',
            in_memory=in_memory,
            **corpusargs,
        )

    def _get_wikiid_wikiname_dict(self, base_folder):

        # collect all wikiids
        wikiid_set = set()
        for data_file in ['aida_conll_testa', 'aida_conll_testb', 'aida_conll_train']:
            with open(base_folder / data_file, mode='r', encoding='utf-8') as read:
                line = read.readline()
                while line:
                    row = line.split('\t')
                    if len(row) > 4:  # line has a wiki annotation
                        wikiid_set.add(row[5].strip())
                    line = read.readline()

        # create the dictionary
        wikiid_wikiname_dict = {}
        wikiid_list = list(wikiid_set)
        ids = ''
        length = len(wikiid_list)

        for i in range(length):
            if (
                    i + 1) % 50 == 0 or i == length - 1:  # there is a limit to the number of ids in one request in the wikimedia api

                ids += wikiid_list[i]
                # request
                resp = requests.get(
                    'https://en.wikipedia.org/w/api.php',
                    params={
                        'action': 'query',
                        'prop': 'info',
                        'pageids': ids,
                        'format': 'json'
                    }
                ).json()

                for wikiid in resp['query']['pages']:
                    try:
                        wikiname = resp['query']['pages'][wikiid]['title'].replace(' ', '_')
                    except KeyError:  # bad wikiid
                        wikiname = 'O'
                    wikiid_wikiname_dict[wikiid] = wikiname
                ids = ''

            else:
                ids += wikiid_list[i]
                ids += '|'

        return wikiid_wikiname_dict


class NEL_ENGLISH_IITB(EntityLinkingCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            in_memory: bool = True,
            ignore_disagreements: bool = False,
            sentence_splitter: SentenceSplitter = SegtokSentenceSplitter(),
            **corpusargs
    ):
        """
        Initialize ITTB Entity Linking corpus introduced in "Collective Annotation of Wikipedia Entities in Web Text" Sayali Kulkarni, Amit Singh, Ganesh Ramakrishnan, and Soumen Chakrabarti.
        If you call the constructor the first time the dataset gets automatically downloaded and transformed in tab-separated column format.

        Parameters
        ----------
        base_path : Union[str, Path], optional
            Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
            to point to a different folder but typically this should not be necessary.
        in_memory: If True, keeps dataset in memory giving speedups in training.
        ignore_disagreements: If True annotations with annotator disagreement will be ignored.
        """
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower() + "_" + type(sentence_splitter).__name__

        # default dataset folder is the cache root
        if not base_path:
            base_path = flair.cache_root / "datasets"
        data_folder = base_path / dataset_name

        iitb_el_docs_path = "https://www.cse.iitb.ac.in/~soumen/doc/CSAW/Annot/CSAW_crawledDocs.tar.gz"
        iitb_el_annotations_path = "https://www.cse.iitb.ac.in/~soumen/doc/CSAW/Annot/CSAW_Annotations.xml"
        corpus_file_name = "iitb.txt"
        parsed_dataset = data_folder / corpus_file_name

        label_type = 'nel'

        if not parsed_dataset.exists():

            docs_zip_path = cached_path(f"{iitb_el_docs_path}", Path("datasets") / dataset_name)
            annotations_xml_path = cached_path(f"{iitb_el_annotations_path}", Path("datasets") / dataset_name)

            unpack_file(docs_zip_path, data_folder, "tar", False)

            import xml.etree.ElementTree as ET
            tree = ET.parse(annotations_xml_path)
            root = tree.getroot()

            # names of raw text documents
            doc_names = set()
            for elem in root:
                doc_names.add(elem[0].text)

            # open output_file
            with open(parsed_dataset, 'w', encoding='utf-8') as write:
                # iterate through all documents
                for doc_name in doc_names:
                    with open(data_folder / 'crawledDocs' / doc_name, 'r', encoding='utf-8') as read:
                        text = read.read()

                        # split sentences and tokenize
                        sentences = sentence_splitter.split(text)
                        sentence_offsets = [sentence.start_pos for sentence in sentences]

                        # iterate through all annotations and add to corresponding tokens
                        for elem in root:

                            if elem[0].text == doc_name and elem[2].text:  # annotation belongs to current document

                                wikiname = elem[2].text.replace(' ', '_')
                                mention_start = int(elem[3].text)
                                mention_length = int(elem[4].text)

                                # find sentence to which annotation belongs
                                sentence_index = 0
                                for i in range(1, len(sentences)):
                                    if mention_start < sentence_offsets[i]:
                                        break
                                    else:
                                        sentence_index += 1

                                # position within corresponding sentence
                                mention_start -= sentence_offsets[sentence_index]
                                mention_end = mention_start + mention_length

                                # set annotation for tokens of entity mention
                                first = True
                                for token in sentences[sentence_index].tokens:
                                    if token.start_pos >= mention_start and token.end_pos <= mention_end:  # token belongs to entity mention
                                        if first:
                                            token.set_label(typename=elem[1].text, value='B-' + wikiname)
                                            first = False
                                        else:
                                            token.set_label(typename=elem[1].text, value='I-' + wikiname)

                        # write to out file
                        write.write('-DOCSTART-\n\n')  # each file is one document

                        for sentence in sentences:

                            for token in sentence.tokens:

                                labels = token.labels

                                if len(labels) == 0:  # no entity
                                    write.write(token.text + '\tO\n')

                                elif len(labels) == 1:  # annotation from one annotator
                                    write.write(token.text + '\t' + labels[0].value + '\n')

                                else:  # annotations from two annotators

                                    if labels[0].value == labels[1].value:  # annotators agree
                                        write.write(token.text + '\t' + labels[0].value + '\n')

                                    else:  # annotators disagree: ignore or arbitrarily take first annotation

                                        if ignore_disagreements:
                                            write.write(token.text + '\tO\n')

                                        else:
                                            write.write(token.text + '\t' + labels[0].value + '\n')

                            write.write('\n')  # empty line after each sentence

        super(NEL_ENGLISH_IITB, self).__init__(
            data_folder,
            train_file=corpus_file_name,
            in_memory=in_memory,
            **corpusargs,
        )


class NEL_ENGLISH_TWEEKI(EntityLinkingCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            in_memory: bool = True,
            **corpusargs,
    ):
        """
        Initialize Tweeki Entity Linking corpus introduced in "Tweeki: Linking Named Entities on Twitter to a Knowledge Graph" Harandizadeh, Singh.
        The data consits of tweets with manually annotated wikipedia links.
        If you call the constructor the first time the dataset gets automatically downloaded and transformed in tab-separated column format.

        Parameters
        ----------
        base_path : Union[str, Path], optional
            Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
            to point to a different folder but typically this should not be necessary.
        in_memory: If True, keeps dataset in memory giving speedups in training.
        """
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = flair.cache_root / "datasets"
        data_folder = base_path / dataset_name

        tweeki_gold_el_path = "https://raw.githubusercontent.com/ucinlp/tweeki/main/data/Tweeki_gold/Tweeki_gold"
        corpus_file_name = "tweeki_gold.txt"
        parsed_dataset = data_folder / corpus_file_name

        # download and parse data if necessary
        if not parsed_dataset.exists():

            original_file_path = cached_path(f"{tweeki_gold_el_path}", Path("datasets") / dataset_name)

            with open(original_file_path, 'r', encoding='utf-8') as read, open(parsed_dataset, 'w',
                                                                               encoding='utf-8') as write:
                line = read.readline()
                while line:
                    if line.startswith('#'):
                        out_line = ''
                    elif line == '\n':  # tweet ends
                        out_line = '\n-DOCSTART-\n\n'
                    else:
                        line_list = line.split('\t')
                        out_line = line_list[1] + '\t'
                        if line_list[3] == '-\n':  # no wiki name
                            out_line += 'O\n'
                        else:
                            out_line += line_list[2][:2] + line_list[3].split('|')[0].replace(' ', '_') + '\n'
                    write.write(out_line)
                    line = read.readline()

            os.rename(original_file_path, str(original_file_path) + '_original')

        super(NEL_ENGLISH_TWEEKI, self).__init__(
            data_folder,
            train_file=corpus_file_name,
            in_memory=in_memory,
            **corpusargs,
        )


class NEL_ENGLISH_REDDIT(EntityLinkingCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            in_memory: bool = True,
            **corpusargs,
    ):
        """
        Initialize the Reddit Entity Linking corpus containing gold annotations only (https://arxiv.org/abs/2101.01228v2) in the NER-like column format.
        The first time you call this constructor it will automatically download the dataset.
        :param base_path: Default is None, meaning that corpus gets auto-downloaded and loaded. You can override this
        to point to a different folder but typically this should not be necessary.
        :param in_memory: If True, keeps dataset in memory giving speedups in training.
        :param document_as_sequence: If True, all sentences of a document are read into a single Sentence object
        """
        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = flair.cache_root / "datasets"
        data_folder = base_path / dataset_name

        # download and parse data if necessary
        reddit_el_path = "https://zenodo.org/record/3970806/files/reddit_el.zip"
        corpus_file_name = "reddit_el_gold.txt"
        parsed_dataset = data_folder / corpus_file_name

        if not parsed_dataset.exists():
            reddit_el_zip = cached_path(f"{reddit_el_path}", Path("datasets") / dataset_name)
            unpack_file(reddit_el_zip, data_folder, "zip", False)

            with open(data_folder / corpus_file_name, "w", encoding='utf-8') as txtout:

                # First parse the post titles
                with open(data_folder / "posts.tsv", "r", encoding='utf-8') as tsvin1, open(
                        data_folder / "gold_post_annotations.tsv", "r", encoding='utf-8') as tsvin2:

                    posts = csv.reader(tsvin1, delimiter="\t")
                    self.post_annotations = csv.reader(tsvin2, delimiter="\t")
                    self.curr_annot = next(self.post_annotations)

                    for row in posts:  # Go through all the post titles

                        txtout.writelines("-DOCSTART-\n\n")  # Start each post with a -DOCSTART- token

                        # Keep track of how many and which entity mentions does a given post title have
                        link_annots = []  # [start pos, end pos, wiki page title] of an entity mention

                        # Check if the current post title has an entity link and parse accordingly
                        if row[0] == self.curr_annot[0]:

                            link_annots.append((int(self.curr_annot[4]), int(self.curr_annot[5]), self.curr_annot[3]))
                            link_annots = self._fill_annot_array(link_annots, row[0], post_flag=True)

                            # Post titles with entity mentions (if any) are handled via this function
                            self._text_to_cols(Sentence(row[2], use_tokenizer=True), link_annots, txtout)
                        else:
                            self._text_to_cols(Sentence(row[2], use_tokenizer=True), link_annots, txtout)

                # Then parse the comments
                with open(data_folder / "comments.tsv", "r", encoding='utf-8') as tsvin3, open(
                        data_folder / "gold_comment_annotations.tsv", "r", encoding='utf-8') as tsvin4:

                    self.comments = csv.reader(tsvin3, delimiter="\t")
                    self.comment_annotations = csv.reader(tsvin4, delimiter="\t")
                    self.curr_annot = next(self.comment_annotations)
                    self.curr_row = next(self.comments)
                    self.stop_iter = False

                    # Iterate over the comments.tsv file, until the end is reached
                    while not self.stop_iter:

                        txtout.writelines("-DOCSTART-\n")  # Start each comment thread with a -DOCSTART- token

                        # Keep track of the current comment thread and its corresponding key, on which the annotations are matched.
                        # Each comment thread is handled as one 'document'.
                        self.curr_comm = self.curr_row[4]
                        comm_key = self.curr_row[0]

                        # Python's csv package for some reason fails to correctly parse a handful of rows inside the comments.tsv file.
                        # This if-condition is needed to handle this problem.
                        if comm_key in {"en5rf4c", "es3ia8j", "es3lrmw"}:
                            if comm_key == "en5rf4c":
                                self.parsed_row = (r.split("\t") for r in self.curr_row[4].split("\n"))
                                self.curr_comm = next(self.parsed_row)
                            self._fill_curr_comment(fix_flag=True)
                        # In case we are dealing with properly parsed rows, proceed with a regular parsing procedure
                        else:
                            self._fill_curr_comment(fix_flag=False)

                        link_annots = []  # [start pos, end pos, wiki page title] of an entity mention

                        # Check if the current comment thread has an entity link and parse accordingly, same as with post titles above
                        if comm_key == self.curr_annot[0]:
                            link_annots.append((int(self.curr_annot[4]), int(self.curr_annot[5]), self.curr_annot[3]))
                            link_annots = self._fill_annot_array(link_annots, comm_key, post_flag=False)
                            self._text_to_cols(Sentence(self.curr_comm, use_tokenizer=True), link_annots, txtout)
                        else:
                            # In two of the comment thread a case of capital letter spacing occurs, which the SegtokTokenizer cannot properly handle.
                            # The following if-elif condition handles these two cases and as result writes full capitalized words in each corresponding row,
                            # and not just single letters into single rows.
                            if comm_key == "dv74ybb":
                                self.curr_comm = " ".join(
                                    [word.replace(" ", "") for word in self.curr_comm.split("  ")])
                            elif comm_key == "eci2lut":
                                self.curr_comm = (self.curr_comm[:18] + self.curr_comm[18:27].replace(" ",
                                                                                                      "") + self.curr_comm[
                                                                                                            27:55] +
                                                  self.curr_comm[55:68].replace(" ", "") + self.curr_comm[
                                                                                           68:85] + self.curr_comm[
                                                                                                    85:92].replace(" ",
                                                                                                                   "") +
                                                  self.curr_comm[92:])

                            self._text_to_cols(Sentence(self.curr_comm, use_tokenizer=True), link_annots, txtout)

        super(NEL_ENGLISH_REDDIT, self).__init__(
            data_folder,
            train_file=corpus_file_name,
            in_memory=in_memory,
            **corpusargs,
        )

    def _text_to_cols(self, sentence: Sentence, links: list, outfile):
        """
        Convert a tokenized sentence into column format
        :param sentence: Flair Sentence object containing a tokenized post title or comment thread
        :param links: array containing information about the starting and ending position of an entity mention, as well
        as its corresponding wiki tag
        :param outfile: file, to which the output is written
        """
        for i in range(0, len(sentence)):
            # If there are annotated entity mentions for given post title or a comment thread
            if links:
                # Keep track which is the correct corresponding entity link, in cases where there is >1 link in a sentence
                link_index = [j for j, v in enumerate(links) if
                              (sentence[i].start_pos >= v[0] and sentence[i].end_pos <= v[1])]
                # Write the token with a corresponding tag to file
                try:
                    if any(sentence[i].start_pos == v[0] and sentence[i].end_pos == v[1] for j, v in enumerate(links)):
                        outfile.writelines(sentence[i].text + "\tS-" + links[link_index[0]][2] + "\n")
                    elif any(
                            sentence[i].start_pos == v[0] and sentence[i].end_pos != v[1] for j, v in enumerate(links)):
                        outfile.writelines(sentence[i].text + "\tB-" + links[link_index[0]][2] + "\n")
                    elif any(
                            sentence[i].start_pos >= v[0] and sentence[i].end_pos <= v[1] for j, v in enumerate(links)):
                        outfile.writelines(sentence[i].text + "\tI-" + links[link_index[0]][2] + "\n")
                    else:
                        outfile.writelines(sentence[i].text + "\tO\n")
                # IndexError is raised in cases when there is exactly one link in a sentence, therefore can be dismissed
                except IndexError:
                    pass

            # If a comment thread or a post title has no entity link, all tokens are assigned the O tag
            else:
                outfile.writelines(sentence[i].text + "\tO\n")

            # Prevent writing empty lines if e.g. a quote comes after a dot or initials are tokenized
            # incorrectly, in order to keep the desired format (empty line as a sentence separator).
            try:
                if ((sentence[i].text in {".", "!", "?", "!*"}) and
                        (sentence[i + 1].text not in {'"', '“', "'", "''", "!", "?", ";)", "."}) and
                        ("." not in sentence[i - 1].text)):
                    outfile.writelines("\n")
            except IndexError:
                # Thrown when the second check above happens, but the last token of a sentence is reached.
                # Indicates that the EOS punctuaion mark is present, therefore an empty line needs to be written below.
                outfile.writelines("\n")

        # If there is no punctuation mark indicating EOS, an empty line is still needed after the EOS
        if sentence[-1].text not in {".", "!", "?"}:
            outfile.writelines("\n")

    def _fill_annot_array(self, annot_array: list, key: str, post_flag: bool) -> list:
        """
        Fills the array containing information about the entity mention annotations, used in the _text_to_cols method
        :param annot_array: array to be filled
        :param key: reddit id, on which the post title/comment thread is matched with its corresponding annotation
        :param post_flag: flag indicating whether the annotations are collected for the post titles (=True)
        or comment threads (=False)
        """
        next_annot = None
        while True:
            # Check if further annotations belong to the current post title or comment thread as well
            try:
                next_annot = next(self.post_annotations) if post_flag else next(self.comment_annotations)
                if next_annot[0] == key:
                    annot_array.append((int(next_annot[4]), int(next_annot[5]), next_annot[3]))
                else:
                    self.curr_annot = next_annot
                    break
            # Stop when the end of an annotation file is reached
            except StopIteration:
                break
        return annot_array

    def _fill_curr_comment(self, fix_flag: bool):
        """
        Extends the string containing the current comment thread, which is passed to _text_to_cols method, when the
        comments are parsed.
        :param fix_flag: flag indicating whether the method is called when the incorrectly imported rows are parsed (=True)
        or regular rows (=False)
        """
        next_row = None
        while True:
            # Check if further annotations belong to the current sentence as well
            try:
                next_row = next(self.comments) if not fix_flag else next(self.parsed_row)
                if len(next_row) < 2:
                    # 'else "  "' is needed to keep the proper token positions (for accordance with annotations)
                    self.curr_comm += next_row[0] if any(next_row) else "  "
                else:
                    self.curr_row = next_row
                    break
            except StopIteration:  # When the end of the comments.tsv file is reached
                self.curr_row = next_row
                self.stop_iter = True if not fix_flag else False
                break