# -*- coding: utf-8 -*-
from __future__ import division
import tkinter as tk
import tkinter.font as tkFont
import tkinter.filedialog as tkFileDialog
import tkinter.constants as Tkconstants
import shutil
import platform
import sys
import re
import string
import glob
import subprocess
import os
from collections import Counter
from threading import Thread
import queue
import string

MODELS = {
    "English": "stanford-corenlp-3.5.1-models.jar",
    "Chinese": "stanford-chinese-corenlp-2015-01-30-models.jar",
    "Spanish": "stanford-spanish-corenlp-2015-12-09-models.jar",
}


def resource_path(relative):
    """Returns the correct path for accessing bundled files."""
    """Use _MEIPASS set by PyInstaller to get the base path"""

    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative)
    return os.path.join(relative)


def start_watcher(def2, count, folder, dataQueue, root):
    t2 = Thread(target=def2, args=(count, folder, dataQueue, root))
    t2.start()


def watcher(count, folder, dataQueue, root):
    import glob
    import time

    counter = 1

    while count > len(glob.glob(folder + "*")):
        # print "Count ", count
        # counter = 1
        if len(glob.glob(folder + "*")) == 0:
            if counter == 1:
                output = "Starting Stanford CoreNLP..."
                counter += 1
            elif counter == 2:
                output = "Starting Stanford CoreNLP."
                counter += 1
            elif counter == 3:
                output = "Starting Stanford CoreNLP.."
                counter += 1
                counter = 1
        else:
            output = (
                "CoreNLP has tagged "
                + str(len(glob.glob(folder + "*")))
                + " of "
                + str(count)
                + " files."
            )
        dataQueue.put(output)
        root.update_idletasks()

        time.sleep(0.3)  # seconds it waits before checking again

    final_message = "CoreNLP has tagged " + str(count) + " of " + str(count) + " files."
    dataQueue.put(output)
    root.update_idletasks()


def call_stan_corenlp_pos(
    class_path,
    language,
    file_list,
    output_folder,
    memory,
    nthreads,
    system,
    dataQueue,
    root,
    parse_type="",
):  # for CoreNLP 3.5.1 (most recent compatible version)
    if language == "English":
        args = "-annotators tokenize,ssplit,pos,lemma" + parse_type
    elif language == "Spanish":
        args = "-props StanfordCoreNLP-spanish.properties"
    elif language == "Chinese":
        args = "-props StanfordCoreNLP-chinese.properties"
    model = MODELS[language]
    # mac osx call:
    if system == "M" or system == "L":
        print(class_path)
        call_parser = (
            "java -cp "
            + class_path
            + "stanford-corenlp-3.5.1.jar:stanford-corenlp-3.5.1-sources.jar:"
            + model
            + ":xom.jar: -Xmx"
            + memory
            + "g edu.stanford.nlp.pipeline.StanfordCoreNLP -threads "
            + nthreads
            + " "
            + args
            + " -filelist "
            + file_list
            + " -outputDirectory "
            + output_folder
            + " -outputFormat xml "
        )
    # windows call:
    elif system == "W":
        call_parser = (
            "java -cp "
            + class_path
            + "*; -Xmx"
            + memory
            + "g edu.stanford.nlp.pipeline.StanfordCoreNLP -threads "
            + nthreads
            + " -annotators tokenize,ssplit,pos,lemma"
            + parse_type
            + " -filelist "
            + file_list
            + " -outputDirectory "
            + output_folder
        )

    count = len(open(file_list, "r").readlines())
    folder = output_folder
    # print "starting checker"
    start_thread(watcher, count, folder, dataQueue, root)

    subprocess.call(
        call_parser, shell=True
    )  # This watches the output folder until all files have been parsed


if platform.system() == "Darwin":
    system = "M"
    title_size = 16
    font_size = 14
    geom_size = "425x450"
    color = "#7AB4C4"
elif platform.system() == "Windows":
    system = "W"
    title_size = 14
    font_size = 12
    geom_size = "460x475"
    color = "#7AB4C4"
elif platform.system() == "Linux":
    system = "L"
    title_size = 14
    font_size = 12
    geom_size = "460x475"
    color = "#7AB4C4"

# This creates a queue in which the core TAALES program can communicate with the GUI
dataQueue = queue.Queue()

# This creates the message for the progress box (and puts it in the dataQueue)
progress = "...Waiting for Data to Process"
dataQueue.put(progress)


def start_thread(def1, *args):
    t = Thread(target=def1, args=args)
    t.start()


class MyApp:
    def __init__(self, parent):

        # Creates font styles - Task: Make these pretty!
        helv14 = tkFont.Font(family="Helvetica", size=font_size)
        helv16 = tkFont.Font(
            family="Helvetica", size=title_size, weight="bold", slant="italic"
        )

        # This defines the GUI parent (ish)
        self.myParent = parent

        # This creates the header text - Task:work with this to make more pretty!
        self.spacer1 = tk.Label(
            parent, text="TAASSC VACS", font=helv16, background=color
        )
        self.spacer1.pack()

        # This creates a frame for the meat of the GUI
        self.thestuff = tk.Frame(parent, background=color)
        self.thestuff.pack()

        # Within the 'thestuff' frame, this creates a frame on the left for the buttons/input
        self.myContainer1 = tk.Frame(self.thestuff, background=color)
        self.myContainer1.pack(side=tk.RIGHT, expand=tk.TRUE)

        # Text to be displayed above the widgets AND for a line to be placed around the elements
        self.labelframe2 = tk.LabelFrame(
            self.myContainer1, text="Instructions", background=color
        )
        self.labelframe2.pack(expand=tk.TRUE)

        # This creates the list of instructions. There may be a better way to do this...
        self.instruct = tk.Label(
            self.labelframe2,
            height="5",
            width="45",
            justify=tk.LEFT,
            padx="4",
            pady="6",
            anchor=tk.W,
            font=helv14,
            text="1. Choose the input folder (where your files are)\n2. Select your output filename\n3. Select the model\n4. Press the 'Process Texts' button",
        )
        self.instruct.pack()

        # Creates Label Frame for Data Input area
        self.secondframe = tk.LabelFrame(
            self.myContainer1, text="Data Input", background=color
        )
        self.secondframe.pack(expand=tk.TRUE)
        # This Places the first button under the instructions.
        self.button1 = tk.Button(self.secondframe)
        self.button1.configure(text="Select Input Folder")
        self.button1.pack()

        # This tells the button what to do when clicked. Currently, only a left-click
        # makes the button do anything (e.g. <Button-1>). The second argument is a "def"
        # That is defined later in the program.
        self.button1.bind("<Button-1>", self.button1Click)

        # Creates default dirname so if statement in Process Texts can check to see
        # if a directory name has been chosen
        self.dirname = ""

        # This creates a label for the first program input (Input Directory)
        self.inputdirlabel = tk.LabelFrame(
            self.secondframe,
            height="1",
            width="45",
            padx="4",
            text="Your selected input folder:",
            background=color,
        )
        self.inputdirlabel.pack()

        # Creates label that informs user which directory has been chosen
        directoryprompt = "(No Folder Chosen)"
        self.inputdirchosen = tk.Label(
            self.inputdirlabel,
            height="1",
            width="44",
            justify=tk.LEFT,
            padx="4",
            anchor=tk.W,
            font=helv14,
            text=directoryprompt,
        )
        self.inputdirchosen.pack()

        # This creates the Output Directory button.
        self.button2 = tk.Button(self.secondframe)
        self.button2["text"] = "Choose Output Filename"
        # This tells the button what to do if clicked.
        self.button2.bind("<Button-1>", self.button2Click)
        self.button2.pack()
        self.outdirname = ""

        # Creates a label for the second program input (Output Directory)
        self.outputdirlabel = tk.LabelFrame(
            self.secondframe,
            height="1",
            width="45",
            padx="4",
            text="Your selected filename:",
            background=color,
        )
        self.outputdirlabel.pack()

        # Creates a label that informs sure which directory has been chosen
        outdirectoryprompt = "(No Output Filename Chosen)"
        self.input2 = ""
        self.outputdirchosen = tk.Label(
            self.outputdirlabel,
            height="1",
            width="44",
            justify=tk.LEFT,
            padx="4",
            anchor=tk.W,
            font=helv14,
            text=outdirectoryprompt,
        )
        self.outputdirchosen.pack()

        # Creates a dropdown menu for model selection
        self.modellabel = tk.LabelFrame(
            self.secondframe,
            height="1",
            width="45",
            padx="4",
            text="Select Model:",
            background=color,
        )
        self.modellabel.pack()
        self.model_var = tk.StringVar(self.modellabel)
        self.model_var.set("English")  # default value
        self.model_menu = tk.OptionMenu(
            self.modellabel, self.model_var, "Chinese", "Spanish", "English"
        )
        self.model_menu.pack()

        self.myContainer2 = tk.Frame(self.secondframe)
        self.myContainer2.pack()

        self.BottomSpace = tk.LabelFrame(
            self.myContainer1, text="Run Program", background=color
        )
        self.BottomSpace.pack()

        self.button3 = tk.Button(self.BottomSpace)
        self.button3["text"] = "Process Texts"
        self.button3.bind("<Button-1>", self.runprogram)
        self.button3.pack()

        self.progresslabelframe = tk.LabelFrame(
            self.BottomSpace, text="Program Status", background=color
        )
        self.progresslabelframe.pack(expand=tk.TRUE)

        # progress = "...Waiting for Data to Process"
        self.progress = tk.Label(
            self.progresslabelframe,
            height="1",
            width="45",
            justify=tk.LEFT,
            padx="4",
            anchor=tk.W,
            font=helv14,
            text=progress,
        )
        self.progress.pack()

        self.poll(self.progress)

    def button1Click(self, event):
        import tkinter.filedialog as tkFileDialog

        self.dirname = tkFileDialog.askdirectory(
            parent=root, title="Please select a directory"
        )
        if self.dirname == "":
            self.displayinputtext = "(No Folder Chosen)"
        else:
            self.displayinputtext = ".../" + self.dirname.split("/")[-1]
        self.inputdirchosen.config(text=self.displayinputtext)

    def button2Click(self, event):
        self.outdirname = tkFileDialog.asksaveasfilename(
            parent=root,
            defaultextension=".txt",
            initialfile="results",
            title="Choose Output Filename",
        )
        if self.outdirname == "":
            self.displayoutputtext = "(No Output Filename Chosen)"
        else:
            self.displayoutputtext = ".../" + self.outdirname.split("/")[-1]
        self.outputdirchosen.config(text=self.displayoutputtext)

    def runprogram(self, event):
        self.poll(self.progress)
        import tkinter.messagebox as tkMessageBox

        if self.dirname == "":
            tkMessageBox.showinfo("Supply Information", "Choose Input Directory")
        if self.outdirname == "":
            tkMessageBox.showinfo("Choose Output Filename", "Choose Output Filename")
        else:
            model = self.model_var.get()
            dataQueue.put("Starting TAASSC...")
            start_thread(
                main,
                model,
                self.dirname,
                self.outdirname,
            )

    def poll(self, function):
        self.myParent.after(10, self.poll, function)
        try:
            function.config(text=dataQueue.get(block=False))
        except queue.Empty:
            pass


def main(language, indir, outfile):
    data_filename = outfile
    data_file = open(data_filename, "w")
    clause_header = [
        "lemma",
        "word",
        "sophistication_vac",
        "filled_sophistication_vac",
        "clause_complexity_vac",
        "filename",
        "sentence_text",
        "position",
    ]
    data_file.write("\t".join(clause_header) + "\n")

    #######

    if not os.path.exists(resource_path("parsed_files/")):
        os.makedirs(resource_path("parsed_files/"))

    if not os.path.exists(resource_path("to_process/")):
        os.makedirs(resource_path("to_process/"))

    folder_list = [resource_path("parsed_files/"), resource_path("to_process/")]

    dataQueue.put("Importing corpus files (this may take a while)...")
    root.update_idletasks()
    for folder in folder_list:
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            os.unlink(file_path)

        copy_files = glob.glob(indir + "/*.txt")

        for thing in copy_files:
            thing_1 = thing
            if system == "M" or system == "L":
                thing = thing.split("/")[-1]
                thing = resource_path("to_process/") + thing
            elif system == "W":
                thing = thing.split("\\")[-1]
                thing = resource_path("to_process\\") + thing
            shutil.copyfile(thing_1, thing)
        input_folder = resource_path("to_process/")

        list_of_files = glob.glob(input_folder + "*.txt")
        ###print "list of files ", list_of_files
        file_list_file = open(input_folder + "_filelist.txt", "w")

        file_list = input_folder + "_filelist.txt"
        ###print "file list ", file_list
        for line in list_of_files:
            line = line + "\n"
            file_list_file.write(line)
        file_list_file.flush()
        file_list_file.close()

    current_directory = resource_path("")
    stan_file_list = input_folder + "_filelist.txt"
    stan_output_folder = resource_path("parsed_files/")
    memory = "3"
    nthreads = "2"
    dataQueue.put("Starting Stanford CoreNLP...")
    root.update_idletasks()

    call_stan_corenlp_pos(
        current_directory,
        language,
        stan_file_list,
        stan_output_folder,
        memory,
        nthreads,
        system,
        dataQueue,
        root,
        parse_type=",depparse",
    )

    try:
        import xml.etree.cElementTree as ET
    except ImportError:
        import xml.etree.ElementTree as ET
    p_files_list = glob.glob(
        resource_path("parsed_files/*.xml")
    )  # Create a list of all files in target folder

    total_nfiles = len(p_files_list)

    nfiles = 0  # This is a counter to see how many files have been processed
    counter = 0

    comp_file_list = []

    verb_tags = "VB VBZ VBP VBD VBN VBG".split(" ")  # This is a list of verb tags
    exclusions = "aux auxpass nsubj dobj iobj amod"

    noun_tags = "NN NNS NNP NNPS VBG".split(
        " "
    )  # note that VBG is included because this list is only used for looking at dependents that will be a nominal
    single_quotes = "u\u2018 u\u2019 u'\u02BC'".split(" ")

    nominals = "NN NNP NNPS NNS PRP PRP$ CD DT".split(" ")
    adjectives = "JJ JJR JJS".split(" ")
    verbs = "VB VBZ VBP VBD VBN VBG".split(" ")
    other = "RB ".split(" ")
    noun_mod = [
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

    for files in p_files_list:
        counter += 1
        nwords = 0
        nsent = 0

        punctuation = ". , ? ! ) ( % / - _ -LRB- -RRB- SYM ".split(" ")

        processed_update = (
            "TAASSC has processed: "
            + str(nfiles)
            + " of "
            + str(total_nfiles)
            + " files."
        )
        dataQueue.put(processed_update)  # output for user
        root.update_idletasks()

        outfilename = files.split("/")[-1]
        outfilename = outfilename.replace(".xml", "")

        comp_file_list.append(outfilename)

        tree = ET.ElementTree(file=files)  # The file is opened by the XML parser

        ###for Clause Complexity and Sophistication

        constructicon = []  # holder for the context-free VACs
        prep_constructicon = []
        lemma_constructicon = []  # holder for the lemma-sensitive VACs
        lemma_constructicon_no_vcop = []  # holder for non-copular constructions
        lemma_constructicon_aux = []  #
        prep_lemma_contructicon = []  #

        #### THE NEXT SECTION CONVERTS THE TREE TO AN APPROXIMATION OF -makeCopulaHead #####
        for sentences in tree.iter("sentence"):
            nsent += 1
            phrase_sentence = []

            noun_list = []
            pronoun_list = []
            for tokens in sentences.iter("token"):
                phrase_sentence.append(tokens[0].text)
                if tokens[4].text in punctuation:
                    continue
                nwords += 1
                if tokens[4].text in noun_tags:
                    noun_list.append(tokens.get("id"))
                if tokens[4].text == "PRP":
                    pronoun_list.append(tokens.get("id"))

            cop_list = (
                []
            )  # list of copular dependency relationships in sentences (tuples)
            cop_list_simple = []

            for deps in sentences.iter("dependencies"):  # iterates through dependencies

                if (
                    deps.get("type") == "collapsed-ccprocessed-dependencies"
                ):  # only iterate through cc-processed-dependencies

                    for dependencies in deps.iter(
                        "dep"
                    ):  # iterate through the dependencies

                        if (
                            dependencies.get("type") == "cop"
                        ):  # if the type is copular...

                            cop_list.append(
                                (
                                    dependencies[0].get("idx"),
                                    dependencies[0].text,
                                    dependencies[1].get("idx"),
                                    dependencies[1].text,
                                )
                            )  # this stores the governor idx and the governor text, and then the dep idx and dep text as a tuple
                            cop_list_simple.append(
                                dependencies[1].get("idx")
                            )  # this should be the id for the copular_verb

                else:
                    sentences.remove(
                        deps
                    )  # this does not get rid of collapsed dependencies. Not sure why.

            for entries in cop_list:
                ##print entries
                comp_check = "no"
                for dependencies in deps.iter("dep"):
                    if (
                        dependencies.get("type") == "cop"
                        and dependencies[0].get("idx") == entries[0]
                    ):  # if the dependency is copular and the item is the one we are concerned with in this iteration:
                        for word in sentences.iter(
                            "token"
                        ):  # iterate through tokens to find the pos tag
                            ##print word[0].text
                            if word.get("id") == entries[0]:
                                pos = word[4].text
                                # nom_comp_position = word.get("id")
                                ##print pos

                        if (
                            pos in nominals
                        ):  # set appropriate relationship (this may be problematic for finite versus non-finite complements)
                            dependencies.set("type", "ncomp")
                            comp_check = "yes"
                        if pos in adjectives:
                            dependencies.set("type", "acomp")
                            comp_check = "yes"
                        if pos in verbs:
                            dependencies.set("type", "vcomp")
                        if pos in other:
                            dependencies.set("type", "other")

                        dependencies[0].set(
                            "idx", entries[2]
                        )  # set the governor as the cop verb
                        dependencies[0].text = entries[
                            3
                        ]  # set the governor as the cop verb
                        dependencies[1].set(
                            "idx", entries[0]
                        )  # set the dependent as the complement
                        dependencies[1].text = entries[
                            1
                        ]  # set the dependent as the complement

                        continue  # if this fixed the comp, continue to the next dependency

                    if (
                        dependencies.get("type") not in noun_mod
                    ):  # if the dependency isn't one that would only work for an nominal (this may need tweaking):

                        if dependencies.get("type") != "tmod" and comp_check == "yes":
                            continue

                        if (
                            dependencies[0].get("idx") == entries[0]
                        ):  # if the governor is the previous cop governor - change to cop
                            dependencies[0].set("idx", entries[2])  # changes idx
                            dependencies[0].text = entries[3]  # changes text

                        if (
                            dependencies[1].get("idx") == entries[0]
                        ):  # if the dependent is the previous cop governor - change to cop
                            dependencies[1].set("idx", entries[2])  # changes idx
                            dependencies[1].text = entries[3]  # changes text

            ### END COPULA CONVERSION SECTION ###

            ### Begin Clause Complexity Section ###

            token_store = []  # This will be a holder of tuples for id, word, lemma, pos
            sentence = []  # stores all of the words so sentence can be stored
            # pos = [] #stores POS information so that it can be easily retrieved later
            verbs = []
            excluded_verbs = []
            gerunds = []
            infinitives = []
            main_verbs = []
            max_sentence_length = 100

            if len(list(sentences.iter("token"))) > max_sentence_length:
                continue

            for tokens in sentences.iter("token"):
                token_store.append(
                    (
                        tokens.get("id"),
                        tokens[0].text.lower(),
                        tokens[1].text,
                        tokens[4].text,
                    )
                )
                sentence.append(tokens[0].text)  # this is word

            inf = "no"
            for items in token_store:
                if items[3] in verb_tags:
                    for dependents in sentences.iter("dependencies"):
                        if (
                            dependents.get("type")
                            == "collapsed-ccprocessed-dependencies"
                        ):
                            for dep in dependents.iter("dep"):
                                if (
                                    dep[1].get("idx") == items[0]
                                    and dep.get("type") in exclusions
                                ):
                                    excluded_verbs.append(
                                        items[0]
                                    )  # adds id to excluded verbs
                if items[3] == "VBG":
                    gerunds.append(
                        items[0]
                    )  # adds id to gerunds (actually any -ing verb)

                if items[3] == "VB" and inf == "yes":
                    infinitives.append(items[0])

                if items[3] == "TO":
                    inf = "yes"
                else:
                    inf = "no"

            for items in token_store:
                if items[0] in excluded_verbs:
                    # print "excluded verb", items[0]
                    continue
                if items[3] in verb_tags:
                    main_verbs.append(items)

            for items in main_verbs:

                if items[0] in cop_list_simple:
                    verb_type = "vcop"
                else:
                    verb_type = "v"

                verb_form = items[2]
                word_entry = items[1].lower()

                if "\xa0" in verb_form:
                    verb_form = verb_form.replace("\xa0", " ")
                for apostrophe in single_quotes:
                    if apostrophe in verb_form:
                        verb_form = verb_form.replace(apostrophe, "'")
                if "-" in verb_form:
                    verb_form = verb_form.replace("-", "_")

                VAC = [
                    [int(items[0]), verb_type, verb_form]
                ]  # format ID, v or v_cop, lemma form

                for dependencies in sentences.iter("dependencies"):
                    if dependencies.get("type") == "collapsed-ccprocessed-dependencies":
                        for dependents in dependencies.iter("dep"):
                            if dependents[0].get("idx") == items[0]:
                                dependent_type = dependents.get(
                                    "type"
                                )  # this allows the program to fix the copula error - nominal complements are now called "ncomp"
                                dependent_id = int(dependents[1].get("idx"))
                                dependent_form = dependents[1].text

                                if dependent_type == "punct":
                                    continue

                                if (
                                    dependent_type == "xcomp"
                                    and token_store[
                                        (int(dependents[1].get("idx")) - 1)
                                    ][3]
                                    in nominals
                                ):
                                    dependent_type = "ncomp"

                                if (
                                    dependent_type == "aux"
                                    and token_store[
                                        (int(dependents[1].get("idx")) - 1)
                                    ][3]
                                    == "MD"
                                ):
                                    dependent_type = "modal"

                                VAC.append(
                                    [dependent_id, dependent_type, dependent_form]
                                )

                VAC = sorted(VAC, key=lambda x: int(x[0]))
                auxilliaries = ["aux", "auxpass", "modal"]
                pre_simple_VAC = []
                simple_VAC = []
                complex_VAC = []
                prep_VAC = []
                simple_VAC_aux = []

                for item in VAC:
                    simple_VAC_aux.append(item[1])
                    if item[1] not in auxilliaries:
                        pre_simple_VAC.append(item)

                # print len(pre_simple_VAC), pre_simple_VAC

                if len(pre_simple_VAC) < 2 and str(pre_simple_VAC[0][0]) in gerunds:
                    # print "g skip"
                    continue

                if len(pre_simple_VAC) < 2 and str(pre_simple_VAC[0][0]) in infinitives:
                    # print "skip"
                    continue

                if len(pre_simple_VAC) < 2 and pre_simple_VAC[0][2] == "be":
                    # print "be skip"
                    continue

                for item in pre_simple_VAC:
                    simple_VAC.append(item[1])
                    complex_VAC.append("_".join([item[1], item[2]]))
                    if "prep" in item[1] and "prepc" not in item[1]:
                        prep_VAC.append("prep")
                    else:
                        prep_VAC.append(item[1])
                simple_VAC_string = "-".join(simple_VAC).lower()
                complex_VAC_string = "-".join(complex_VAC).lower()

                if "-v_be-" in complex_VAC_string:
                    complex_VAC_string = complex_VAC_string.replace(
                        "-v_be-", "-vcop_be-"
                    )
                    simple_VAC_string = simple_VAC_string.replace("-v-", "-vcop-")

                sentence_string = " ".join(sentence)
                database_string = (
                    "\t".join(
                        [
                            verb_form.lower(),
                            word_entry,
                            simple_VAC_string,
                            complex_VAC_string,
                            "-".join(simple_VAC_aux).lower(),
                            outfilename,
                            sentence_string,
                        ]
                    )
                    + "\n"
                )
                database_string_clean = filter(
                    lambda x: x in string.printable, database_string
                )
                data_file.write(database_string_clean)

        nfiles += 1  # add to counter

        data_file.flush()
        data_file.close()

    ######Clean-up:
    folder_list = [resource_path("parsed_files/"), resource_path("to_process/")]

    for folder in folder_list:
        if os.path.exists(folder):
            for the_file in os.listdir(folder):
                file_path = os.path.join(folder, the_file)
                os.unlink(file_path)
    #######

    finishmessage = "Processed " + str(nfiles) + " Files"
    dataQueue.put(finishmessage)
    root.update_idletasks()
    if system == "M":
        # self.progress.config(text =finishmessage)
        import tkinter.messagebox as tkMessageBox

        tkMessageBox.showinfo("Finished!", "Your files have been processed by TAASSC")


class Catcher:
    def __init__(self, func, subst, widget):
        self.func = func
        self.subst = subst
        self.widget = widget

    def __call__(self, *args):
        try:
            if self.subst:
                args = self.subst(*args)
            return self.func(*args)
        except SystemExit as msg:
            raise SystemExit(msg)
        except Exception:
            import traceback
            import tkinter.messagebox as tkMessageBox

            ermessage = traceback.format_exc(1)
            ermessage = re.sub(r".*(?=Error)", "", ermessage, flags=re.DOTALL)
            ermessage = "There was a problem processing your files:\n\n" + ermessage
            tkMessageBox.showerror("Error Message", ermessage)


if __name__ == "__main__":
    root = tk.Tk()
    root.wm_title("TAASSC 1.3.8")
    root.configure(background=color)
    # sets starting size: NOTE: it doesn't appear as though Tkinter will let you make the
    # starting size smaller than the space needed for widgets.
    root.geometry(geom_size)
    tk.CallWrapper = Catcher
    myapp = MyApp(root)
    root.mainloop()
