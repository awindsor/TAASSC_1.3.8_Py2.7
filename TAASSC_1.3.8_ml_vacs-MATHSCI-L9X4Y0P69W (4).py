# -*- coding: utf-8 -*-
from __future__ import division
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLineEdit,
)
import logging
from PySide6.QtCore import Qt, QThread, Signal
from SizedQueue import SizedQueue
import multiprocessing
import shutil
import sys
import os
import time
import subprocess
import glob
from collections import namedtuple

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def setup_logging(log_level, log_queue):
    """Configure logging in the main process to consume from the queue"""
    assert log_level in [
        logging.DEBUG,
        logging.INFO,
        logging.WARNING,
        logging.ERROR,
        logging.CRITICAL,
    ], "Invalid log level"

    logger = logging.getLogger()
    handler = logging.StreamHandler()  # Print logs to console
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(log_level)

    queue_listener = logging.handlers.QueueListener(log_queue, handler)
    queue_listener.start()
    return queue_listener


HEADER = (
    "\t".join(
        [
            "lemma",
            "word",
            "sophistication_vac",
            "filled_sophistication_vac",
            "clause_complexity_vac",
            "filename",
            "sentence_number",
            "sentence_text",
        ]
    )
    + "\n"
)


MODELS = {
    "English": (
        "",
        "-annotators tokenize,ssplit,pos,lemma,parse",
    ),
    "Chinese": (
        "stanford-chinese-corenlp-2015-01-30-models.jar",
        "-props StanfordCoreNLP-chinese.properties " + "-segment.verbose false",
    ),
    "Spanish": (
        "stanford-spanish-corenlp-2015-12-09-models.jar",
        "-props StanfordCoreNLP-spanish.properties",
    ),
}

MEMORY = "12"
NTHREADS = multiprocessing.cpu_count() - 2
SYSTEM = "M" if sys.platform == "darwin" else "W" if sys.platform == "win32" else "L"

VERB_TAGS = "VB VBZ VBP VBD VBN VBG".split(" ")  # This is a list of verb tags
CHINESE_VERB_TAGS = "VV VC VE VA".split(" ")
# VA is an adjective used without a copula in Chinese
# VC 是 (shì) is a copula between subject and predicate, equivalent to "be" in English
# VE 有 (yǒu) is a verb that means “to have” or “to exist.”
# VV is a general verb tag

EXCLUSIONS = "aux auxpass nsubj dobj iobj amod"
# Verbs that are the dependents of following dependency types are excluded from
# verb argument constructions because they do not represent the core predicate:
#
# 1. aux: Auxiliary verbs support the main verb by forming tenses, aspects, or
#    moods but don't carry the central semantic load.
#    E.g., "She has finished her homework." → "has" is an aux.
#
# 2. auxpass: Passive auxiliaries indicate the passive voice; they help form
#    the verb phrase without being the main predicate.
#    E.g., "The cake was eaten by the children." → "was" is auxpass.
#
# 3. nsubj: Nominal subjects (often gerunds) function as noun phrases rather
#    than as independent predicates.
#    E.g., "Singing is her passion." → "Singing" is the nsubj functioning as a
#    noun.
#
# 4. dobj: Direct objects that are verb forms (commonly gerunds) serve as
#    arguments of another verb rather than acting as the central predicate.
#    E.g., "I enjoy swimming." → "swimming" is the dobj.
#
# 5. iobj: Indirect objects, when formed by verb derivatives, function as
#    arguments and do not establish the main action.
#    E.g., "The teacher gave dancing a try." → "dancing" is the iobj.
#
# 6. amod: Adjectival modifiers (often participles derived from verbs) describe
#    nouns instead of operating as the main predicate.
#    E.g., "He comforted the crying baby." → "crying" is an amod.

CHINESE_EXCLUSIONS = "aux nsubj dobj amod"

NOUN_TAGS = "NN NNS NNP NNPS VBG".split(" ")
# note that VBG is included because this list is only used for looking at dependents that will be a nominal

CHINESE_NOUN_TAGS = "NN NR NT PN".split(" ")
SINGLE_QUOTES = "u\u2018 u\u2019 u'\u02BC'".split(" ")

NOMINAL_TAGS = "NN NNP NNPS NNS PRP PRP$ CD DT".split(" ")
CHINESE_NOMIAL_TAGS = "NN NR NT PN".split(" ")

ADJECTIVES = "JJ JJR JJS".split(" ")
CHINESE_ADJECTIVES = "JJ".split(" ")
OTHER_TAGS = "RB ".split(" ")
NOUN_MODS = [
    "amod",
    "appos",
    "det",
    "goeswith",
    "mwe",
    "nn",
    "num",
    "poss",
    "cop",
    "advmod",
    "advcl",
    "rcmod",
    "vmod",
]  # note: cop is thrown in for convenience; #advmod and advcl added in .8.5 , "advmod", "advcl"
PUNCTUATION = ". , ? ! ) ( % / - _ -LRB- -RRB- SYM ".split(" ")
AUXILLIARIES = ["aux", "auxpass", "modal"]

MAX_SENTENCE_LENGTH = 300

Token = namedtuple("Token", ["id", "word", "lemma", "pos"])


def resource_path(relative):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative)
    return os.path.join(relative)


# We use threads within the GUI code to handle watching all the subprocesses
class WorkerThread(QThread):
    update = Signal(str)  # Signal to send updates

    def __init__(self, func, *args):
        super().__init__()
        self.func = func
        self.args = args

    def run(self):
        self.func(*self.args)


class WatcherThreadStatus(QThread):
    update_signal = Signal(str)

    def __init__(self, status_queue):
        super().__init__()
        self.status_queue = status_queue

    def run(self):
        while True:
            msg = self.status_queue.get()
            if msg is None:
                break
            self.update_signal.emit(msg)


class WatcherThreadStan(QThread):
    update_signal = Signal(str)

    def __init__(self, output_folder, file_count):
        super().__init__()
        self.output_folder = output_folder
        self.file_count = file_count

    def run(self):
        processed_files = 0
        prev_processed_files = None
        while processed_files < self.file_count:
            processed_files = len(os.listdir(self.output_folder))
            if prev_processed_files is None or prev_processed_files < processed_files:
                msg = (
                    f"CoreNLP has tagged {processed_files} of {self.file_count} files."
                )
                self.update_signal.emit(msg)
                prev_processed_files = processed_files
            time.sleep(1)
        msg = f"CoreNLP has tagged {self.file_count} of {self.file_count} files."
        self.update_signal.emit(msg)


def dependency_parse_gui(file_list, output_folder, update_signal, call_parser):
    file_count = sum(1 for _ in open(file_list, "r"))
    # Because the processing happens in a different process, we need to determine
    # progress by watching the files appear in the output folder.
    update_signal.emit("CoreNLP is tagging files...")
    watcher_thread = WatcherThreadStan(output_folder, file_count)
    watcher_thread.update_signal.connect(update_signal)
    watcher_thread.start()
    watcher_thread.finished.connect(watcher_thread.deleteLater)
    subprocess.run(call_parser, shell=True, check=True)
    watcher_thread.quit()
    watcher_thread.wait()


def get_parser_for_system(
    class_path, file_list, output_folder, memory, nthreads, system, model, args
):
    jar_files = [
        "stanford-corenlp-3.5.1.jar",
        "stanford-corenlp-3.5.1-models.jar",
        "xom.jar",
    ]
    jar_files += [model]
    jar_files = [os.path.join(class_path, jar) for jar in jar_files]
    connector = (
        ";" if system == "W" else ":"
    )  # Windows uses semicolon, others use colon
    jar_files = connector.join(jar_files)
    call_parser = (
        f"java -cp '{jar_files}' -Xmx{memory}g "
        f"edu.stanford.nlp.pipeline.StanfordCoreNLP -threads {nthreads} {args} "
        f"-filelist {file_list} -outputDirectory {output_folder} -outputFormat xml"
    )
    return call_parser


def delete_files(folder):
    """Delete all files in a folder."""
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        os.unlink(file_path)


def prepare_stan_dirs(
    text_file_dir, stan_output_folder, stan_input_folder, file_list_path
):

    os.makedirs(stan_output_folder, exist_ok=True)
    delete_files(stan_output_folder)
    os.makedirs(stan_input_folder, exist_ok=True)
    delete_files(stan_input_folder)

    with open(file_list_path, "w") as file_list:
        for txt_file in glob.glob(os.path.join(text_file_dir, "*.txt")):
            shutil.copy(txt_file, stan_input_folder)
            file_list.write(txt_file + "\n")


def call_corenlp(
    class_path,
    language,
    file_list,
    output_folder,
    memory,
    nthreads,
    system,
    update_signal,
):
    assert system in ["M", "W", "L"], "Invalid system type."

    """Executes Stanford CoreNLP for dependency parsing."""

    model, args = MODELS[language]
    call_parser = get_parser_for_system(
        class_path, file_list, output_folder, memory, nthreads, system, model, args
    )
    dependency_parse_gui(file_list, output_folder, update_signal, call_parser)


class ProcessingThread(QThread):
    progress_signal = Signal(str)

    def __init__(
        self, language, text_file_dir, outfilename, retain_xml=False, xml_dir=False
    ):
        super().__init__()
        self.retain_xml = retain_xml
        self.xml_dir = xml_dir
        self.language = language
        self.text_file_dir = text_file_dir
        self.outfilename = outfilename

    def run(self):
        self.progress_signal.emit("Processing started...")

        if self.isInterruptionRequested():
            return
        main(
            self.language,
            self.text_file_dir,
            self.outfilename,
            self.progress_signal,
            retain_xml=self.retain_xml,
            xml_dir=self.xml_dir,
        )
        self.progress_signal.emit("Processing completed.")

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TAASSC VACS")
        self.setFixedSize(450, 550)
        self.layout = QVBoxLayout()
        self.input_path = ""
        self.output_file = ""
        self.thread = None
        self.setup_ui()
        self.setLayout(self.layout)

    def setup_ui(self):
        self.title_label = QLabel("TAASSC VACS")
        self.layout.addWidget(self.title_label, alignment=Qt.AlignCenter)

        # Input Selection
        self.input_label = QLabel("Input Path (Directory or CSV File):")
        self.layout.addWidget(self.input_label)
        self.input_layout = QHBoxLayout()
        self.input_textbox = QLineEdit()
        self.input_button = QPushButton("Browse")
        self.input_button.clicked.connect(self.select_input_path)
        self.input_layout.addWidget(self.input_textbox)
        self.input_layout.addWidget(self.input_button)
        self.layout.addLayout(self.input_layout)

        # Directory Type Dropdown
        self.directory_type_label = QLabel("Directory Type:")
        self.layout.addWidget(self.directory_type_label)
        self.directory_type_dropdown = QComboBox()
        self.directory_type_dropdown.addItems(["XML Files", "TXT Files"])
        self.layout.addWidget(self.directory_type_dropdown)

        # ID Column Selection
        self.id_column_label = QLabel("ID Column:")
        self.layout.addWidget(self.id_column_label)
        self.id_column_dropdown = QComboBox()
        self.id_column_dropdown.currentIndexChanged.connect(self.validate_columns)
        self.layout.addWidget(self.id_column_dropdown)

        # Text Column Selection
        self.text_column_label = QLabel("Text Column:")
        self.layout.addWidget(self.text_column_label)
        self.text_column_dropdown = QComboBox()
        self.text_column_dropdown.currentIndexChanged.connect(self.validate_columns)
        self.layout.addWidget(self.text_column_dropdown)

        # Retain XML Checkbox
        self.retain_xml_checkbox = QCheckBox("Retain XML")
        self.layout.addWidget(self.retain_xml_checkbox)

        # Language Dropdown
        self.language_label = QLabel("Select Language:")
        self.layout.addWidget(self.language_label)
        self.language_dropdown = QComboBox()
        self.language_dropdown.addItems(["English", "Chinese", "Spanish"])
        self.layout.addWidget(self.language_dropdown)

        # Output File Selection
        self.output_label = QLabel("Output File (.tsv):")
        self.layout.addWidget(self.output_label)
        self.output_layout = QHBoxLayout()
        self.output_textbox = QLineEdit()
        self.output_button = QPushButton("Browse")
        self.output_button.clicked.connect(self.select_output_file)
        self.output_layout.addWidget(self.output_textbox)
        self.output_layout.addWidget(self.output_button)
        self.layout.addLayout(self.output_layout)

        # Program Updates Text Area
        self.progress_frame = QFrame()
        self.progress_frame.setFrameShape(QFrame.Box)
        self.progress_frame.setLineWidth(1)
        self.progress_layout = QVBoxLayout(self.progress_frame)
        self.progress_text = QTextEdit()
        self.progress_text.setReadOnly(True)
        self.progress_layout.addWidget(self.progress_text)
        self.layout.addWidget(self.progress_frame)

        # Process Button
        self.process_button = QPushButton("Process Input")
        self.process_button.setEnabled(False)
        self.layout.addWidget(self.process_button, alignment=Qt.AlignCenter)
        self.process_button.clicked.connect(self.start_processing)

        # Set Initial States
        self.directory_type_dropdown.setEnabled(False)
        self.id_column_dropdown.setEnabled(False)
        self.text_column_dropdown.setEnabled(False)
        self.retain_xml_checkbox.setEnabled(False)

    def validate_columns(self):
        if self.id_column_dropdown.currentText() == self.text_column_dropdown.currentText():
            self.progress_text.append("Error: ID Column and Text Column must be distinct.")
            self.process_button.setEnabled(False)
        else:
            self.process_button.setEnabled(True)

    def start_processing(self):
        self.process_button.setEnabled(False)
        self.thread = ProcessingThread(self.language_dropdown.currentText(), self.input_path, self.output_file, self.text_column_dropdown.currentText())
        self.thread.progress_signal.connect(self.progress_text.append)
        self.thread.start()

    def closeEvent(self, event):
        if self.thread and self.thread.isRunning():
            self.thread.requestInterruption()
            self.thread.wait()
        event.accept()

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TAASSC VACS")
        self.setFixedSize(425, 500)
        self.layout = QVBoxLayout()
        self.input_path = ""
        self.output_file = ""
        self.thread = None
        self.setup_ui()
        self.setLayout(self.layout)

    def setup_ui(self):
        self.title_label = QLabel("TAASSC VACS")
        self.layout.addWidget(self.title_label, alignment=Qt.AlignCenter)

        self.input_label = QLabel("Input Path (Directory or CSV File):")
        self.layout.addWidget(self.input_label)
        self.input_layout = QHBoxLayout()
        self.input_textbox = QLineEdit()
        self.input_button = QPushButton("Browse")
        self.input_button.clicked.connect(self.select_input_path)
        self.input_layout.addWidget(self.input_textbox)
        self.input_layout.addWidget(self.input_button)
        self.layout.addLayout(self.input_layout)

        # File type selection
        self.input_type_dropdown = QComboBox()
        self.input_type_dropdown.addItems(["Directory", "CSV File"])
        self.input_type_dropdown.currentIndexChanged.connect(self.toggle_text_column)
        self.layout.addWidget(self.input_type_dropdown)

        # Text column selection (only for CSV files)
        self.text_column_label = QLabel("Text Column:")
        self.text_column_input = QLineEdit()
        self.layout.addWidget(self.text_column_label)
        self.layout.addWidget(self.text_column_input)
        self.text_column_label.setEnabled(False)
        self.text_column_input.setEnabled(False)

        # Output selection
        self.output_label = QLabel("Output File:")
        self.layout.addWidget(self.output_label)
        self.output_layout = QHBoxLayout()
        self.output_textbox = QLineEdit()
        self.output_button = QPushButton("Browse")
        self.output_button.clicked.connect(self.select_output_file)
        self.output_layout.addWidget(self.output_textbox)
        self.output_layout.addWidget(self.output_button)
        self.layout.addLayout(self.output_layout)

        # Language selection
        self.language_label = QLabel("Select Language:")
        self.layout.addWidget(self.language_label)
        self.language_dropdown = QComboBox()
        self.language_dropdown.addItems(["English", "Spanish", "Chinese"])
        self.layout.addWidget(self.language_dropdown)

        # Progress display
        self.progress_frame = QFrame()
        self.progress_frame.setFrameShape(QFrame.Box)
        self.progress_frame.setLineWidth(1)
        self.progress_layout = QVBoxLayout(self.progress_frame)
        self.progress_label = QLabel("Program Status: ...Waiting for Data to Process")
        self.progress_layout.addWidget(self.progress_label)
        self.layout.addWidget(self.progress_frame)

        self.process_button = QPushButton("Process Text")
        self.process_button.setFixedWidth(150)
        self.layout.addWidget(self.process_button, alignment=Qt.AlignCenter)
        self.process_button.clicked.connect(self.start_processing)

    def toggle_text_column(self):
        if self.input_type_dropdown.currentText() == "CSV File":
            self.text_column_label.setEnabled(True)
            self.text_column_input.setEnabled(True)
        else:
            self.text_column_label.setEnabled(False)
            self.text_column_input.setEnabled(False)
            self.text_column_input.clear()

    def select_input_path(self):
        if self.input_type_dropdown.currentText() == "Directory":
            directory = QFileDialog.getExistingDirectory(self, "Select Input Directory")
            if directory:
                self.input_path = directory
                self.input_textbox.setText(directory)
        else:
            file, _ = QFileDialog.getOpenFileName(
                self, "Select CSV File", "", "CSV Files (*.csv)"
            )
            if file:
                self.input_path = file
                self.input_textbox.setText(file)

    def select_output_file(self):
        file, _ = QFileDialog.getSaveFileName(
            self, "Select Output File", "", "Text Files (*.txt)"
        )
        if file:
            self.output_file = file
            self.output_textbox.setText(file)

    def update_progress(self, message):
        self.progress_label.setText(f"Program Status: {message}")
        logger.info(message)
        if message == "Processing completed.":
            self.process_button.setEnabled(True)

    def start_processing(self):
        self.input_path = self.input_textbox.text().strip()
        self.output_file = self.output_textbox.text().strip()

        if not self.input_path or not self.output_file:
            self.update_progress(
                "Error: Please select or enter input path and output file."
            )
            return

        text_column = (
            self.text_column_input.text().strip()
            if self.input_type_dropdown.currentText() == "CSV File"
            else None
        )
        self.process_button.setEnabled(False)
        language = self.language_dropdown.currentText()
        self.thread = ProcessingThread(
            language, self.input_path, self.output_file, text_column
        )
        self.thread.progress_signal.connect(self.update_progress)
        self.thread.start()

    def closeEvent(self, event):
        if self.thread and self.thread.isRunning():
            self.thread.requestInterruption()
            self.thread.wait()
        event.accept()


def process_xml_chinese(filename, output_queue):
    """
    Find all the VACS in the parsed document. VACs are operationalized by Kyle
    as a verb and all its immediate descendants. Initially we implement this with
    essentially no modification
    """
    logger.debug(f"Processing {filename} using Chinese VAC")
    tree = ET.ElementTree(file=filename)  # Parse the XML file.

    # Iterate over each sentence element in the document.
    for sent_num, sentence in enumerate(tree.iter("sentence"), start=1):
        # Skip sentences that exceed the maximum allowed length.
        if len(list(sentence.iter("token"))) > MAX_SENTENCE_LENGTH:
            continue

        # Construct a data structure (token_store) that holds each token's details,
        # along with a list of all sentence words and a mapping of token IDs to their POS tags.
        token_store, sentence_words, pos_list = construct_token_store(sentence)

        # ----------------------------------------------
        # Dependency Extraction: Collapsed & CC-Processed
        # ----------------------------------------------
        # Extract the dependency tree that has been processed to collapse certain relations
        # (such as prepositions and conjunctions) for a more compact representation.
        dependency = get_collapsed_ccprocessed_dependency(sentence)
        # Ensure there is exactly one dependency tree of this type per sentence.

        # ----------------------------------------------
        # Copula (Copular) Constructions Adjustment
        # ----------------------------------------------
        # In many languages and grammatical analyses, the predicate is built around
        # the main verb. In a copular construction (e.g., "The sky is blue"), the
        # complement (e.g., "blue") is linked to the copular verb ("is") via a "cop"
        # dependency relation with "blue" as governor and "is" as dependent. However,
        # for verb argument construction extraction, we want the copular verb to serve
        # as the head of the construction. This adjustment reassigns the dependency
        # roles so that all arguments are attached to the verb, and it also reclassifies
        # the type of complement (nominal, adjectival, verbal, etc.) based on the part
        # of speech tag.

        # Identify all copular dependency relationships in the sentence. Each tuple
        # holds:
        #   (complement candidate index, complement text, copular verb index, copular
        #   verb text)
        dependency, cop_verb_ids = fix_copula(pos_list, dependency)

        # ----------------------------------------------
        # End Copula Conversion Section
        # ----------------------------------------------
        # Note: In languages such as Chinese where the copula is often implicit (i.e., not overtly expressed),
        # this adjustment may be unnecessary and therefore can be skipped.
        # ----------------------------------------------
        # Begin Clause Complexity and Verb Argument Construction
        # ----------------------------------------------
        # Initialize lists to store various verb types:
        # - excluded_verbs: verbs that occur in dependency relations (e.g., auxiliaries) we wish to ignore.
        # - gerunds: tokens tagged as -ing forms (VBG) which often act as modifiers rather than main verbs.
        #     As a noun: “Running is good exercise.” (subject) "I enjoy swimming." (object) "Her favorite activity is painting." (predicate noun)
        #   Note that VGB catches not only gerunds but also present participles.
        # - A present participle functions as an adjective or as part of a continuous verb tense: "The crying baby needed comfort." "The baby is crying."
        # - infinitives: base verbs that follow a "TO" marker. Infinitive verbs do not act as verbs but can
        #     As a noun: (as subject) “To err is human.” (as object) “She likes to read.”
        # 	  As an adjective: “He has a paper to write.”
        # 	  As an adverb: “They went to the store to buy groceries.”
        # - main_verbs: verbs that remain after exclusions.
        excluded_verbs = []
        gerunds = []
        infinitives = []
        main_verbs = []

        inf = "no"  # A flag to indicate if the current context is an infinitive (after a "TO" marker).
        # Iterate over each token in the sentence.
        for token in token_store:
            # For tokens with verb POS tags, check if they are in an excluded dependency relation.
            if token.pos in VERB_TAGS:
                for dep in dependency.iter("dep"):
                    # If the token appears as a dependent in a relation that is marked for exclusion,
                    # add its ID to the list of excluded verbs.
                    if dep[1].get("idx") == token.id and dep.get("type") in EXCLUSIONS:
                        excluded_verbs.append(token.id)
            # Identify gerunds (e.g., "running", "swimming") by the VBG tag.
            if token.pos == "VBG":
                gerunds.append(token.id)

            # Identify infinitives: a token tagged VB that immediately follows a "TO".
            if token.pos == "VB" and inf == "yes":
                infinitives.append(token.id)

            # Set the infinitive flag when encountering a "TO" token.
            if token.pos == "TO":
                inf = "yes"
            else:
                inf = "no"

        # Determine the main verbs by selecting those that are verbs and not marked for exclusion.
        main_verbs = [
            token
            for token in token_store
            if token.pos in VERB_TAGS and token.id not in excluded_verbs
        ]

        # ----------------------------------------------
        # Process Each Main Verb to Build Its Argument Construction
        # ----------------------------------------------
        for token in main_verbs:
            # Label the verb: if it appears as a copular verb, mark as "vcop"; otherwise, "v".
            verb_type = "vcop" if token.id in cop_verb_ids else "v"
            verb_lemma = token.lemma  # Retrieve the base (lemma) form of the verb.
            word_entry = (
                token.word
            )  # The original word form as it appeared in the sentence.

            # Clean up the verb lemma (e.g., remove extra morphological markers).
            verb_lemma = clean_verb_lemma(verb_lemma)

            # Initialize the verb argument construction (VAC) with the verb itself.
            # The format is [token ID, verb type, verb lemma].
            VAC = [[int(token.id), verb_type, verb_lemma]]

            # Iterate over every dependency relation in the sentence.
            for dep in dependency.iter("dep"):
                # Find dependencies where the current token is the governor (i.e., the head of the relation).
                if dep[0].get("idx") == token.id:
                    # Extract the dependency relation type (e.g., subject, object, complement).
                    dependent_type = dep.get("type")
                    dependent_id = int(dep[1].get("idx"))
                    dependent_form = dep[
                        1
                    ].text  # The actual word form of the dependent.

                    # Skip punctuation dependencies.
                    if dependent_type == "punct":
                        continue

                    # For open clausal complements ("xcomp") that actually function as nominal complements,
                    # reassign the dependency label to "ncomp".
                    if (
                        dependent_type == "xcomp"
                        and token_store[(int(dep[1].get("idx")) - 1)].pos
                        in NOMINAL_TAGS
                    ):
                        dependent_type = "ncomp"

                    # For auxiliary dependencies that serve a modal function, reassign the label to "modal".
                    if (
                        dependent_type == "aux"
                        and token_store[(int(dep[1].get("idx")) - 1)].pos == "MD"
                    ):
                        dependent_type = "modal"

                    # Append this dependent’s information to the VAC.
                    VAC.append([dependent_id, dependent_type, dependent_form])

            # Sort the VAC by the token IDs to maintain the original left-to-right order in the sentence.
            VAC = sorted(VAC, key=lambda x: int(x[0]))

            # Prepare various lists to create simplified and complex representations of the VAC.
            pre_simple_VAC = (
                []
            )  # A filtered list that excludes auxiliary relationships.
            simple_VAC = []  # A list of dependency labels (simplified).
            complex_VAC = (
                []
            )  # A list of combined dependency labels and the dependent word forms.
            prep_VAC = []  # Special handling for prepositional dependencies.
            simple_VAC_aux = []  # A list of all dependency labels (for reference).

            # Populate the simplified representation by filtering out auxiliary labels.
            for item in VAC:
                simple_VAC_aux.append(item[1])
                if item[1] not in AUXILLIARIES:
                    pre_simple_VAC.append(item)

            # If the resulting argument construction is too trivial (less than two elements)
            # and the only element is either a gerund, an infinitive, or the verb "be",
            # then skip processing this verb as it may not represent a full argument structure.
            if len(pre_simple_VAC) < 2 and str(pre_simple_VAC[0][0]) in gerunds:
                continue
            if len(pre_simple_VAC) < 2 and str(pre_simple_VAC[0][0]) in infinitives:
                continue
            if len(pre_simple_VAC) < 2 and pre_simple_VAC[0][2] == "be":
                continue

            # Build the final simple and complex representations of the verb argument construction.
            for item in pre_simple_VAC:
                simple_VAC.append(
                    item[1]
                )  # Use only the dependency type for the simple version.
                complex_VAC.append(
                    "_".join([item[1], item[2]])
                )  # Combine the label with the actual word.
                # Normalize prepositional dependencies to a standard "prep" label.
                if "prep" in item[1] and "prepc" not in item[1]:
                    prep_VAC.append("prep")
                else:
                    prep_VAC.append(item[1])
            simple_VAC_string = "-".join(simple_VAC).lower()
            complex_VAC_string = "-".join(complex_VAC).lower()

            # Correct any remaining copula representations: if the pattern for a "be" verb is detected,
            # change the label from "v_be" to "vcop_be" to indicate the copular function.
            if "-v_be-" in complex_VAC_string:
                complex_VAC_string = complex_VAC_string.replace("-v_be-", "-vcop_be-")
                simple_VAC_string = simple_VAC_string.replace("-v-", "-vcop-")

            # Construct the final output string that represents the verb argument construction.
            # This string includes:
            # 1. The cleaned verb lemma.
            # 2. The original verb word form.
            # 3. The simple VAC (sequence of dependency types).
            # 4. The complex VAC (dependency type + word form).
            # 5. The full sequence of dependency types.
            # 6. The file identifier (derived from the filename).
            # 7. The sentence number.
            # 8. The full sentence text.
            database_string = (
                "\t".join(
                    [
                        verb_lemma.lower(),  # canonical form of the verb
                        word_entry,  # original verb form
                        simple_VAC_string,  # simple verb argument construction
                        complex_VAC_string,  # detailed verb argument construction
                        "-".join(
                            simple_VAC_aux
                        ).lower(),  # full sequence of dependency labels
                        os.path.basename(filename).replace(
                            ".xml", ""
                        ),  # identifier for the source file
                        str(sent_num),  # sentence number within the document
                        " ".join(sentence_words),  # complete sentence text
                    ]
                )
                + "\n"
            )
            # Place the constructed string into an output queue for later processing (e.g., database storage).
            output_queue.put(database_string)


def get_collapsed_ccprocessed_dependency(sentence):
    cc_dependencies = [
        dependency
        for dependency in sentence.iter("dependencies")
        if dependency.get("type") == "collapsed-ccprocessed-dependencies"
    ]
    assert (
        len(cc_dependencies) == 1
    ), "There should be exactly one collapsed-ccprocessed-dependencies element in each sentence."
    dependency = cc_dependencies[0]
    return dependency


def process_xml(filename, output_queue):
    """Find all the VACS in the parsed document. VACs are operationalized by Kyle
    as a verb and all its immediate descendants.
    """
    logger = logging.getLogger("parse_xml")
    queue_handler = logging.handlers.QueueHandler(log_queue)  # Send logs to the queue
    logger.addHandler(queue_handler)
    logger.setLevel(logging.DEBUG)
    tree = ET.ElementTree(file=filename)  # Parse the XML file.

    # Iterate over each sentence element in the document.
    for sent_num, sentence in enumerate(tree.iter("sentence"), start=1):
        # Skip sentences that exceed the maximum allowed length.
        if len(list(sentence.iter("token"))) > MAX_SENTENCE_LENGTH:
            continue

        # Construct a data structure (token_store) that holds each token's details,
        # along with a list of all sentence words and a mapping of token IDs to their POS tags.
        token_store, sentence_words, pos_list = construct_token_store(sentence)

        # ----------------------------------------------
        # Dependency Extraction: Collapsed & CC-Processed
        # ----------------------------------------------
        # Extract the dependency tree that has been processed to collapse certain relations
        # (such as prepositions and conjunctions) for a more compact representation.
        dependency = get_cc_collapsed_dependency(sentence)

        # ----------------------------------------------
        # Copula (Copular) Constructions Adjustment
        # ----------------------------------------------
        # In many languages and grammatical analyses, the predicate is built around
        # the main verb. In a copular construction (e.g., "The sky is blue"), the
        # complement (e.g., "blue") is linked to the copular verb ("is") via a "cop"
        # dependency relation with "blue" as governor and "is" as dependent. However,
        # for verb argument construction extraction, we want the copular verb to serve
        # as the head of the construction. This adjustment reassigns the dependency
        # roles so that all arguments are attached to the verb, and it also reclassifies
        # the type of complement (nominal, adjectival, verbal, etc.) based on the part
        # of speech tag.
        cop_verb_ids = fix_copula(pos_list, dependency)  # changes dependency in place

        # ----------------------------------------------
        # End Copula Conversion Section
        # ----------------------------------------------
        # Note: In languages such as Chinese where the copula is often implicit (i.e., not overtly expressed),
        # this adjustment may be unnecessary and therefore can be skipped.

        # ----------------------------------------------
        # Begin Verb Argument Construction
        # ----------------------------------------------
        # Initialize lists to store various verb types:
        # - excluded_verbs:
        #       Verbs that occur in dependency relations (e.g., auxiliaries) we wish to ignore.
        # - gerunds: tokens tagged as -ing forms (VBG) which often act as modifiers rather than main verbs.
        #     As a noun: “Running is good exercise.” (subject) "I enjoy swimming." (object) "Her favorite activity is painting." (predicate noun)
        #   Note that VGB catches not only gerunds but also present participles.
        # - A present participle functions as an adjective or as part of a continuous verb tense: "The crying baby needed comfort." "The baby is crying."
        # - infinitives: base verbs that follow a "TO" marker. Infinitive verbs do not act as verbs but can
        #     As a noun: (as subject) “To err is human.” (as object) “She likes to read.”
        # 	  As an adjective: “He has a paper to write.”
        # 	  As an adverb: “They went to the store to buy groceries.”
        # - main_verbs: verbs that remain after exclusions.
        excluded_verbs = []
        gerunds = []
        infinitives = []
        main_verbs = []

        infinitive = "no"  # A flag to indicate if the current context is an infinitive (after a "TO" marker).
        # Iterate over each token in the sentence.
        for token in token_store:
            # For tokens with verb POS tags, check if they are in an excluded dependency relation.
            if token.pos in VERB_TAGS:
                for dep in dependency.iter("dep"):
                    # If the token appears as a dependent in a relation that is marked for exclusion,
                    # add its ID to the list of excluded verbs.
                    if dep[1].get("idx") == token.id and dep.get("type") in EXCLUSIONS:
                        excluded_verbs.append(token.id)
            # Identify gerunds (e.g., "running", "swimming") by the VBG tag.
            if token.pos == "VBG":
                gerunds.append(token.id)

            # Identify infinitives: a token tagged VB that immediately follows a "TO".
            if token.pos == "VB" and infinitive == "yes":
                infinitives.append(token.id)

            # Set the infinitive flag when encountering a "TO" token.
            if token.pos == "TO":
                infinitive = "yes"
            else:
                infinitive = "no"

        # Determine the main verbs by selecting those that are verbs and not marked for exclusion.
        main_verbs = [
            token
            for token in token_store
            if token.pos in VERB_TAGS and token.id not in excluded_verbs
        ]

        # ----------------------------------------------
        # Process Each Main Verb to Build Its Argument Construction
        # ----------------------------------------------
        for token in main_verbs:
            # Label the verb: if it appears as a copular verb, mark as "vcop"; otherwise, "v".
            verb_type = "vcop" if token.id in cop_verb_ids else "v"
            verb_lemma = token.lemma  # Retrieve the base (lemma) form of the verb.
            word_entry = (
                token.word
            )  # The original word form as it appeared in the sentence.

            # Clean up the verb lemma (e.g., remove extra morphological markers).
            verb_lemma = clean_verb_lemma(verb_lemma)

            # Initialize the verb argument construction (VAC) with the verb itself.
            # The format is [token ID, verb type, verb lemma].
            VAC = [[int(token.id), verb_type, verb_lemma]]

            # Iterate over every dependency relation in the sentence.
            for dep in dependency.iter("dep"):
                # Find dependencies where the current token is the governor (i.e., the head of the relation).
                if dep[0].get("idx") == token.id:
                    # Extract the dependency relation type (e.g., subject, object, complement).
                    dependent_type = dep.get("type")
                    dependent_id = int(dep[1].get("idx"))
                    dependent_form = dep[
                        1
                    ].text  # The actual word form of the dependent.

                    # Skip punctuation dependencies.
                    if dependent_type == "punct":
                        continue

                    # For open clausal complements ("xcomp") that actually function as nominal complements,
                    # reassign the dependency label to "ncomp".
                    if (
                        dependent_type == "xcomp"
                        and token_store[(int(dep[1].get("idx")) - 1)].pos
                        in NOMINAL_TAGS
                    ):
                        dependent_type = "ncomp"

                    # For auxiliary dependencies that serve a modal function, reassign the label to "modal".
                    if (
                        dependent_type == "aux"
                        and token_store[(int(dep[1].get("idx")) - 1)].pos == "MD"
                    ):
                        dependent_type = "modal"

                    # Append this dependent’s information to the VAC.
                    VAC.append([dependent_id, dependent_type, dependent_form])

            # Sort the VAC by the token IDs to maintain the original left-to-right order in the sentence.
            VAC = sorted(VAC, key=lambda x: int(x[0]))

            # Prepare various lists to create simplified and complex representations of the VAC.
            pre_simple_VAC = (
                []
            )  # A filtered list that excludes auxiliary relationships.
            simple_VAC = []  # A list of dependency labels (simplified).
            complex_VAC = (
                []
            )  # A list of combined dependency labels and the dependent word forms.
            prep_VAC = []  # Special handling for prepositional dependencies.
            simple_VAC_aux = []  # A list of all dependency labels (for reference).

            # Populate the simplified representation by filtering out auxiliary labels.
            for item in VAC:
                simple_VAC_aux.append(item[1])
                if item[1] not in AUXILLIARIES:
                    pre_simple_VAC.append(item)

            # If the resulting argument construction is too trivial (less than two elements)
            # and the only element is either a gerund, an infinitive, or the verb "be",
            # then skip processing this verb as it may not represent a full argument structure.
            if len(pre_simple_VAC) < 2 and str(pre_simple_VAC[0][0]) in gerunds:
                continue
            if len(pre_simple_VAC) < 2 and str(pre_simple_VAC[0][0]) in infinitives:
                continue
            if len(pre_simple_VAC) < 2 and pre_simple_VAC[0][2] == "be":
                continue

            # Build the final simple and complex representations of the verb argument construction.
            for item in pre_simple_VAC:
                simple_VAC.append(
                    item[1]
                )  # Use only the dependency type for the simple version.
                complex_VAC.append(
                    "_".join([item[1], item[2]])
                )  # Combine the label with the actual word.
                # Normalize prepositional dependencies to a standard "prep" label.
                if "prep" in item[1] and "prepc" not in item[1]:
                    prep_VAC.append("prep")
                else:
                    prep_VAC.append(item[1])
            simple_VAC_string = "-".join(simple_VAC).lower()
            complex_VAC_string = "-".join(complex_VAC).lower()

            # Correct any remaining copula representations: if the pattern for a "be" verb is detected,
            # change the label from "v_be" to "vcop_be" to indicate the copular function.
            if "-v_be-" in complex_VAC_string:
                complex_VAC_string = complex_VAC_string.replace("-v_be-", "-vcop_be-")
                simple_VAC_string = simple_VAC_string.replace("-v-", "-vcop-")

            # Construct the final output string that represents the verb argument construction.
            # This string includes:
            # 1. The cleaned verb lemma.
            # 2. The original verb word form.
            # 3. The simple VAC (sequence of dependency types).
            # 4. The complex VAC (dependency type + word form).
            # 5. The full sequence of dependency types.
            # 6. The file identifier (derived from the filename).
            # 7. The sentence number.
            # 8. The full sentence text.
            database_string = (
                "\t".join(
                    [
                        verb_lemma.lower(),  # canonical form of the verb
                        word_entry,  # original verb form
                        simple_VAC_string,  # simple verb argument construction
                        complex_VAC_string,  # detailed verb argument construction
                        "-".join(
                            simple_VAC_aux
                        ).lower(),  # full sequence of dependency labels
                        os.path.basename(filename).replace(
                            ".xml", ""
                        ),  # identifier for the source file
                        str(sent_num),  # sentence number within the document
                        " ".join(sentence_words),  # complete sentence text
                    ]
                )
                + "\n"
            )
            # Place the constructed string into an output queue for later processing (e.g., database storage).
            output_queue.put(database_string)


def get_cc_collapsed_dependency(sentence):
    cc_dependencies = [
        dependency
        for dependency in sentence.iter("dependencies")
        if dependency.get("type") == "collapsed-ccprocessed-dependencies"
    ]
    # Ensure there is exactly one dependency tree of this type per sentence.
    assert (
        len(cc_dependencies) == 1
    ), "There should be exactly one collapsed-ccprocessed-dependencies element in each sentence."
    dependency = cc_dependencies[0]
    return dependency


def fix_copula(pos_list, dependency):
    logger.debug("Fixing copulas")
    logger.debug(ET.tostring(dependency))
    cop_list = [
        (
            dep[0].get(
                "idx"
            ),  # Index of the complement candidate (currently attached as governor)
            dep[0].text,  # The actual word/text of the complement candidate
            dep[1].get(
                "idx"
            ),  # Index of the copular verb (currently attached as dependent)
            dep[1].text,  # The text of the copular verb
        )
        for dep in dependency.iter("dep")
        if dep.get("type")
        == "cop"  # Filter: only process dependencies labeled as 'cop'
    ]

    # Keep track of all copular verb IDs for later reference (if needed).
    cop_verb_ids = list(map(lambda x: x[2], cop_list))

    # For each identified copular construction, adjust the dependencies.
    for entry in cop_list:
        comp_check = (
            "no"  # A flag indicating if we've already updated a complement role.
        )

        # Iterate over every dependency in the sentence to find those involving the copular construction.
        for dep in dependency.iter("dep"):
            # Check if this dependency is a copular link for the current construction.
            if dep.get("type") == "cop" and dep[0].get("idx") == entry[0]:
                # Retrieve the part-of-speech (POS) tag for the complement candidate.
                pos = pos_list[entry[0]]

                # Reassign the dependency relation based on the POS of the complement:
                # - If the complement is nominal (e.g., a noun), label it as "ncomp".
                if pos in NOMINAL_TAGS:
                    dep.set("type", "ncomp")
                    comp_check = "yes"
                    # - If the complement is adjectival (e.g., an adjective), label it as "acomp".
                if pos in ADJECTIVES:
                    dep.set("type", "acomp")
                    comp_check = "yes"
                    # - If the complement is verbal (e.g., a verb form), label it as "vcomp".
                if pos in VERB_TAGS:
                    dep.set("type", "vcomp")
                    # - For any other POS categories, use a generic "other" label.
                if pos in OTHER_TAGS:
                    dep.set("type", "other")

                    # **Linguistic Motivation:**
                    # By reassigning the dependency type, we capture the nature of the complement in a way
                    # that’s consistent with how other arguments are labeled. This makes downstream processing
                    # (such as pattern extraction for verb argument constructions) more uniform.

                    # Swap the roles in the dependency so that the copular verb becomes the governor (head):
                    # - The complement token now gets the index and text of the copular verb.
                dep[0].set("idx", entry[2])
                dep[0].text = entry[3]
                # - The copular verb (originally the governor) now takes the complement’s original index.
                dep[1].set("idx", entry[0])
                dep[1].text = entry[1]

                # Once the primary copular dependency is fixed, move to the next dependency.
                continue

                # For dependencies not exclusively associated with nominal modifiers:
            if dep.get("type") not in NOUN_MODS:
                # If the dependency is not a temporal modifier ("tmod") and we've already fixed the complement,
                # skip further adjustments for efficiency.
                if dep.get("type") != "tmod" and comp_check == "yes":
                    continue

                    # In case there are other dependencies in the sentence that refer to the original complement
                    # (which was previously the head of some relations), update them so that they now refer to
                    # the copular verb. This ensures that the entire dependency structure consistently uses
                    # the copular verb as the head of its argument structure.

                    # If the governor in the dependency is the original complement, update it.
                if dep[0].get("idx") == entry[0]:
                    dep[0].set("idx", entry[2])
                    dep[0].text = entry[3]

                    # Likewise, if the dependent is the original complement, update it as well.
                if dep[1].get("idx") == entry[0]:
                    dep[1].set("idx", entry[2])
                    dep[1].text = entry[3]
    logger.debug("After copula repair")
    logger.debug(ET.tostring(dependency))
    return cop_verb_ids


def construct_token_store(sentence):

    token_store = []
    sentence_words = []
    pos_dict = {}
    for token in sentence.iter("token"):
        id = token.get("id")
        word = token.find("word").text
        if token.find("lemma") is not None:
            lemma = token.find("lemma").text
        else:
            lemma = word.lower()
        pos = token.find("POS").text
        pos_dict[id] = pos
        token_store.append(Token(id, word.lower(), lemma, pos))
        sentence_words.append(word)
    return token_store, sentence_words, pos_dict


def clean_verb_lemma(verb_lemma):
    if "\xa0" in verb_lemma:
        verb_lemma = verb_lemma.replace("\xa0", " ")
    for apostrophe in SINGLE_QUOTES:
        if apostrophe in verb_lemma:
            verb_lemma = verb_lemma.replace(apostrophe, "'")
    if "-" in verb_lemma:
        verb_lemma = verb_lemma.replace("-", "_")
    return verb_lemma


def process_queue(input_queue, output_queue, status_queue, log_queue, nfiles, language):
    """
    Worker function that reads from the input queue, processes the data, and writes to the output queue.
    There will be multiple of these.
    """
    while True:
        filename = input_queue.get()
        if filename is None:  # Stop signal
            break
        if language == "Chinese":
            logger.debug(
                f"Processing: {filename} using Chinese VAC extractor. {input_queue.qsize()} files left."
            )
            process_xml_chinese(filename, output_queue, log_queue)
        else:
            logger.debug(
                f"Processing: {filename} using VAC extractor. {input_queue.qsize()} files left."
            )
            process_xml(filename, output_queue, status_queue,log_queue)
        status_queue.put(
            f"Program Status: Processed {filename}.\nProcessed {nfiles - input_queue.qsize()} of {nfiles} files."
        )


def write_output(output_filename, output_queue):
    """Writer function that reads from the output queue and writes to a single CSV. There will be one of these."""
    logger.debug(f"Writing to {output_filename}")
    with open(
        output_filename,
        "w",
    ) as output_file:
        output_file.write(HEADER)
        while True:
            row = output_queue.get()
            if row is None:  # Stop signal
                break
            output_file.write(row)


def build_queues(p_files_list):
    input_queue = SizedQueue()
    output_queue = multiprocessing.Queue()
    status_queue = multiprocessing.Queue()
    log_queue = multiprocessing.Queue()
    for file in p_files_list:
        input_queue.put(file)  # Create a list of all files in target folder
    logger.debug(f"Built input queue with {input_queue.qsize()} files.")
    return input_queue, output_queue, status_queue, log_queue


def move_xml_files(stan_output_folder, outfilename):
    out_dir = os.path.dirname(outfilename)
    for xml_file in glob.glob(os.path.join(stan_output_folder, "*.xml")):
        new_xml_file = os.path.basename(xml_file).replace(".xml", "_parsed.xml")
        shutil.move(
            xml_file,
            os.path.join(out_dir, new_xml_file),
        )

def clean_temp_files(stan_output_folder, stan_input_folder):
    delete_folder_list = [stan_output_folder, stan_input_folder]
    for folder in delete_folder_list:
        delete_files(folder)


def start_workers(input_queue, output_queue, status_queue, nfiles, language):
    workers = []
    for _ in range(min(NTHREADS, nfiles)):
        p = multiprocessing.Process(
            target=process_queue,
            args=(input_queue, output_queue, status_queue, nfiles, language),
        )
        p.start()
        workers.append(p)
    return workers


def start_writer(outfilename, output_queue):
    writer_process = multiprocessing.Process(
        target=write_output, args=(outfilename, output_queue)
    )
    writer_process.start()
    return writer_process


def start_watcher(update_status, status_queue):
    watcher_thread = WatcherThreadStatus(status_queue)
    watcher_thread.update_signal.connect(update_status)
    watcher_thread.start()
    watcher_thread.finished.connect(watcher_thread.deleteLater)
    return watcher_thread


def stop_watcher_thread(status_queue, watcher_thread):
    status_queue.put(None)  # Signal watcher to stop
    watcher_thread.quit()
    watcher_thread.wait()


def stop_workers(input_queue, workers):
    for _ in workers:
        input_queue.put(None, increment=False)  # Send termination signal
    for p in workers:
        p.join(timeout=10)  # Ensure all workers finish


def stop_writer(output_queue, writer_process):
    output_queue.put(None)  # Signal writer to stop
    writer_process.join(timeout=10)  # Ensure writer finishes


def main(
    language, input_dir, outfilename, update_status, xml_dir=False, retain_xml=False
):
    """Main processing function."""
    try:
        if not xml_dir:
            stan_output_folder = resource_path("parsed_files/")
            stan_input_folder = resource_path("to_process/")
            current_directory = resource_path("")
            file_list_path = os.path.join(stan_input_folder, "_filelist.txt")

            update_status.emit("Preparing Stanford NLP directories...")
            logger.info("Preparing Stanford NLP directories...")
            prepare_stan_dirs(
                input_dir, stan_output_folder, stan_input_folder, file_list_path
            )

            update_status.emit("Calling CoreNLP for processing...")
            logger.info("Calling CoreNLP for processing...")
            call_corenlp(
                current_directory,
                language,
                file_list_path,
                stan_output_folder,
                MEMORY,
                NTHREADS,
                SYSTEM,
                update_status,
            )
            p_files_list = glob.glob(resource_path("parsed_files/*.xml"))
        else:
            p_files_list = glob.glob(input_dir + "/*.xml")
        update_status.emit("Building processing queues...")
        logger.info("Building processing queues...")
        input_queue, output_queue, status_queue,log_queue= build_queues(p_files_list)
        nfiles = input_queue.qsize()

        watcher_thread = start_watcher(update_status, status_queue)
        writer_process = start_writer(outfilename, output_queue)
        workers = start_workers(
            input_queue, output_queue, status_queue, nfiles, language
        )
        stop_workers(input_queue, workers)
        stop_writer(output_queue, writer_process)
        stop_watcher_thread(status_queue, watcher_thread)
        update_status.emit("Cleaning up temporary files...")
        if retain_xml:
            move_xml_files(stan_output_folder, outfilename)
        if not xml_dir:
            clean_temp_files(
                stan_output_folder,
                stan_input_folder,
            )
        while not status_queue.empty():
            time.sleep(1)
        finish_message = f"Processed {nfiles} Files"
        update_status.emit(finish_message)

    except Exception as e:
        update_status.emit(f"Error occurred: {e}")
        logger.exception(f"An error occurred: {e}")

    finishmessage = "Processed " + str(nfiles) + " Files"
    update_status.emit(finishmessage)

    # if SYSTEM == "M":
    #     msg_box = QMessageBox()
    #     msg_box.setIcon(QMessageBox.Information)
    #     msg_box.setWindowTitle("Finished!")
    #     msg_box.setText("Your files have been processed by TAASSC")
    #     msg_box.exec()


# class Catcher:
#     def __init__(self, func, subst, widget):
#         self.func = func
#         self.subst = subst
#         self.widget = (
#             widget  # A QWidget instance to serve as the parent for QMessageBox
#         )

#     def __call__(self, *args):
#         try:
#             if self.subst:
#                 args = self.subst(*args)
#             return self.func(*args)
#         except SystemExit as msg:
#             raise SystemExit(msg)
#         except Exception:
#             ermessage = traceback.format_exc()
#             ermessage = re.sub(r".*(?=Error)", "", ermessage, flags=re.DOTALL)
#             ermessage = "There was a problem processing your files:\n\n" + ermessage

#             # Show error message in Qt
#             QMessageBox.critical(self.widget, "Error Message", ermessage)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myapp = MyApp()
    myapp.show()
    app.exec()
