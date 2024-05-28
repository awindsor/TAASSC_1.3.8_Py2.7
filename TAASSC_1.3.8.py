#-*- coding: utf-8 -*-
from __future__ import division
import Tkinter as tk
import tkFont
import tkFileDialog
import Tkconstants
import shutil
import platform
import sys 
import re
import string
import glob
import math
import subprocess
import os
from collections import Counter
from threading import Thread
import Queue
import string

import csv

def resource_path(relative):
	if hasattr(sys, "_MEIPASS"):
		return os.path.join(sys._MEIPASS, relative)
	return os.path.join(relative)

ktatk = resource_path("ktatk.py")
import ktatk as ktk

#"#0091CC"
if platform.system() == "Darwin":
	system = "M"
	title_size = 16
	font_size = 14
	geom_size = "425x575"
	color = "#7AB4C4"
elif platform.system() == "Windows":
	system = "W"
	title_size = 14
	font_size = 12
	geom_size = "460x600"
	color = "#7AB4C4"
elif platform.system() == "Linux":
	system = "L"
	title_size = 14
	font_size = 12
	geom_size = "460x600"
	color = "#7AB4C4"

#This creates a que in which the core TAALES program can communicate with the GUI
dataQueue = Queue.Queue()

#This creates the message for the progress box (and puts it in the dataQueue)
progress = "...Waiting for Data to Process"
dataQueue.put(progress)

def start_thread(def1, arg1, arg2, arg3):
	t = Thread(target=def1, args=(arg1, arg2, arg3))
	t.start()

class MyApp:
	def __init__(self, parent):
		
		#Creates font styles - Task: Make these pretty!
		
		
		helv14= tkFont.Font(family= "Helvetica", size=font_size)
		#times14= tkFont.Font(family= "Times", size=14)
		helv16= tkFont.Font(family= "Helvetica", size = title_size, weight = "bold", slant = "italic")
		
		#This defines the GUI parent (ish)
		self.myParent = parent
		
		#This creates the header text - Task:work with this to make more pretty!
		self.spacer1= tk.Label(parent, text= "Tool for the Automatic Analysis of Syntactic\nSophistication and Complexity", font = helv16, background = color)
		self.spacer1.pack()
		
		#This creates a frame for the meat of the GUI
		self.thestuff= tk.Frame(parent, background =color)
		self.thestuff.pack()
		#Currently, the sizes aren't doing anything...

		#Within the 'thestuff' frame, this creates a frame on the left for the buttons/input
		self.myContainer1 = tk.Frame(self.thestuff, background = color)
		self.myContainer1.pack(side = tk.RIGHT, expand = tk.TRUE)


		#Text to be displayed above the widgets AND for a line to be placed around the elements
		self.labelframe2 = tk.LabelFrame(self.myContainer1, text= "Instructions", background = color)
		self.labelframe2.pack(expand=tk.TRUE)
	

				
		#This creates the list of instructions.	 There may be a better way to do this...
		self.instruct = tk.Label(self.labelframe2, height = "5", width = "45", justify = tk.LEFT, padx = "4", pady= "6", anchor = tk.W, font = helv14, text ="1. Select desired index types\n2. If desired, select text output type(s)\n3. Choose the input folder (where your files are)\n4. Select your output filename\n5. Press the 'Process Texts' button")
		self.instruct.pack()
		
		self.checkboxframe = tk.LabelFrame(self.myContainer1, text= "Options", background = color, width = "45")
		self.checkboxframe.pack(expand=tk.TRUE)

		self.sca_box = tk.IntVar()			
		self.cb2 = tk.Checkbutton(self.checkboxframe, text="L2SCA\nVariables", variable=self.sca_box,background = color)
		self.cb2.grid(row=1,column=4, sticky = "W")		
		self.cb2.deselect()
		
		self.clause_box = tk.IntVar()
		self.cb3 = tk.Checkbutton(self.checkboxframe, text="Clause\nComplexity", variable=self.clause_box,background = color)
		self.cb3.grid(row=1,column=1, sticky = "W")	
		self.cb3.select()
		
		self.phrase_box = tk.IntVar()
		self.cb4 = tk.Checkbutton(self.checkboxframe, text="Phrase\nComplexity", variable=self.phrase_box,background = color)
		self.cb4.grid(row=1,column=2, sticky = "W")	
		self.cb4.select()
		
		self.sophistication_box = tk.IntVar()
		self.cb5 = tk.Checkbutton(self.checkboxframe, text="Syntactic\nSophistication", variable=self.sophistication_box,background = color)
		self.cb5.grid(row=1,column=3, sticky = "W")	
		self.cb5.select()

		self.components_box = tk.IntVar()
		self.cb6 = tk.Checkbutton(self.checkboxframe, text="Syntactic\nComponents", variable=self.components_box,background = color)
		self.cb6.grid(row=2,column=1, sticky = "W")	
		self.cb6.deselect()

		self.freqframe = tk.LabelFrame(self.myContainer1, text= "VAC Frequency List Generator", background = color, width = "45")
		self.freqframe.pack(expand=tk.TRUE)

		self.generate_box = tk.IntVar()
		self.generate = tk.Checkbutton(self.freqframe, text="Create\nNew List", variable=self.generate_box,background = color)
		self.generate.grid(row=1,column=1, sticky = "W")	
		self.generate.deselect()

		self.freq_label = tk.Label(self.freqframe, text = "Minimum VAC\nFrequency:",background = color)
		self.freq_label.grid(row=1,column=2, sticky = "W")
			
		self.min_freq = tk.Spinbox(self.freqframe,from_=1, to = 10,width = 2)
		self.min_freq.grid(row=1,column=3, sticky = "W")		
					
		self.database_box = tk.IntVar()
		self.cb7 = tk.Checkbutton(self.checkboxframe, text="Output\nText", variable=self.database_box,background = color)
		self.cb7.grid(row=2,column=2, sticky = "W")	
		self.cb7.deselect()
		
		self.database_box_2 = tk.IntVar()
		self.cb7 = tk.Checkbutton(self.checkboxframe, text="Output\nXML", variable=self.database_box_2,background = color)
		self.cb7.grid(row=2,column=3, sticky = "W")	
		self.cb7.deselect()
		
		self.var_list = [self.sca_box,self.clause_box,self.phrase_box,self.sophistication_box,self.components_box,self.database_box,self.database_box_2,self.generate_box,self.min_freq]

		#Creates Label Frame for Data Input area
		self.secondframe= tk.LabelFrame(self.myContainer1, text= "Data Input", background = color)
		self.secondframe.pack(expand=tk.TRUE) 
		#This Places the first button under the instructions.
		self.button1 = tk.Button(self.secondframe)
		self.button1.configure(text= "Select Input Folder")
		self.button1.pack()
		
		#This tells the button what to do when clicked.	 Currently, only a left-click
		#makes the button do anything (e.g. <Button-1>). The second argument is a "def"
		#That is defined later in the program.
		self.button1.bind("<Button-1>", self.button1Click)
		
		#Creates default dirname so if statement in Process Texts can check to see
		#if a directory name has been chosen
		self.dirname = ""
		
		#This creates a label for the first program input (Input Directory)
		self.inputdirlabel =tk.LabelFrame(self.secondframe, height = "1", width= "45", padx = "4", text = "Your selected input folder:", background = color)
		self.inputdirlabel.pack()
		
		#Creates label that informs user which directory has been chosen
		directoryprompt = "(No Folder Chosen)"
		self.inputdirchosen = tk.Label(self.inputdirlabel, height= "1", width= "44", justify=tk.LEFT, padx = "4", anchor = tk.W, font= helv14, text = directoryprompt)
		self.inputdirchosen.pack()
		
		#This creates the Output Directory button.
		self.button2 = tk.Button(self.secondframe)
		self.button2["text"]= "Choose Output Filename"
		#This tells the button what to do if clicked.
		self.button2.bind("<Button-1>", self.button2Click)
		self.button2.pack()
		self.outdirname = ""
		
		#Creates a label for the second program input (Output Directory)
		self.outputdirlabel = tk.LabelFrame(self.secondframe, height = "1", width= "45", padx = "4", text = "Your selected filename:", background = color)
		self.outputdirlabel.pack()
		
		#Creates a label that informs sure which directory has been chosen
		#outdirectoryprompt = "(No output Folder Chosen)"
		outdirectoryprompt = "(No Output Filename Chosen)"
		self.input2 = ""
		self.outputdirchosen = tk.Label(self.outputdirlabel, height= "1", width= "44", justify=tk.LEFT, padx = "4", anchor = tk.W, font= helv14, text = outdirectoryprompt)
		self.outputdirchosen.pack()
		
		self.myContainer2 = tk.Frame(self.secondframe)
		self.myContainer2.pack()
				
		self.BottomSpace= tk.LabelFrame(self.myContainer1, text = "Run Program", background = color)
		self.BottomSpace.pack()

		self.button3= tk.Button(self.BottomSpace)
		self.button3["text"] = "Process Texts"
		self.button3.bind("<Button-1>", self.runprogram)
		self.button3.pack()
		
		#self.spacer2 = Label(self.BottomSpace, text="\n", background = color)
		#self.spacer2.pack()

		self.progresslabelframe = tk.LabelFrame(self.BottomSpace, text= "Program Status", background = color)
		self.progresslabelframe.pack(expand= tk.TRUE)
		
		#progress = "...Waiting for Data to Process"
		self.progress= tk.Label(self.progresslabelframe, height= "1", width= "45", justify=tk.LEFT, padx = "4", anchor = tk.W, font= helv14, text=progress)
		self.progress.pack()
		
		self.poll(self.progress)
		
	#not currently used		
	def cb_all_Click(self, event):
		for items in self.box_list[:9]:
			items.select()

	def cb_none_Click(self, event):
		for items in self.box_list[:9]:
			items.deselect()
			
	
	#Following is an example of how we can update the information from users...
	def button1Click(self, event):
		#import Tkinter, 
		import tkFileDialog
		self.dirname = tkFileDialog.askdirectory(parent=root,title='Please select a directory')
		#print self.dirname
		if self.dirname == "":
			self.displayinputtext = "(No Folder Chosen)"
		else: 
			self.displayinputtext = '.../'+self.dirname.split('/')[-1]
		self.inputdirchosen.config(text = self.displayinputtext)
		

	def button2Click(self, event):
		#self.outdirname = tkFileDialog.askdirectory(parent=root,initialdir="/",title='Please select a directory')
		self.outdirname = tkFileDialog.asksaveasfilename(parent=root, defaultextension = ".csv", initialfile = "results",title='Choose Output Filename')
		#print self.outdirname
		if self.outdirname == "":
			self.displayoutputtext = "(No Output Filename Chosen)"
		else: self.displayoutputtext = '.../' + self.outdirname.split('/')[-1]
		self.outputdirchosen.config(text = self.displayoutputtext)
		
		
	def runprogram(self, event):
		self.poll(self.progress)
		import tkMessageBox
		if self.dirname is "":
			tkMessageBox.showinfo("Supply Information", "Choose Input Directory")
		if self.outdirname is "":
			tkMessageBox.showinfo("Choose Output Filename", "Choose Output Filename")
		if self.var_list[4].get() == 1:
			if self.var_list[1].get() == 0 or self.var_list[2].get() == 0 or self.var_list[3].get() == 0:
				tkMessageBox.showinfo("Component Score Error", "To calculate syntactic components, you must also calculate clause complexity, phrase complexity, and syntactic sophistication indices.\n\nPlease check these boxes.")
				comp_checker = "not ok"
			else: comp_checker = "ok"
		else: comp_checker = "ok"
		if self.dirname is not "" and self.outdirname is not "" and comp_checker == "ok":
		
			dataQueue.put("Starting TAASSC...")
			start_thread(main, self.dirname, self.outdirname,self.var_list)
			
	def poll(self, function):
		
		self.myParent.after(10, self.poll, function)
		try:
			function.config(text = dataQueue.get(block=False))
			
			#root.update_idletasks()
		except Queue.Empty:
			pass

def main(indir, outfile,check_list):

	####################################
	#This function deals with denominator issues that can kill the program:
	## TODO: This should return some form of NAN instead of 0. 
	def safe_divide(numerator, denominator):
		if denominator == 0:
			index = 0
		else: index = numerator/denominator
		return index
	####################################
	def dict_counter(dictionary, structure, denominator):
		n_structure = dictionary[structure]
		index_count = n_structure
		if isinstance(denominator, int) == True:
			denom_count = denominator
		else: denom_count = len(denominator)

		index = safe_divide(index_count, denom_count)
	
		return index
	#########################################
	def dict_builder(database_file): #builds dictionaries from database files
		lemma_freq_dict ={}
		construction_freq_dict = {}
		contingency_dict ={}	
		for entries in database_file:  
			entries = entries.split("\t")
			if len(entries)<2:
				continue
			lemma_freq_dict[entries[0]]=entries[2]
			construction_freq_dict[entries[1]]=entries[3]
			combined_key = entries[0] + "\t" + entries[1]
			contingency_dict[combined_key] = entries[4:]

		return_list = [lemma_freq_dict,construction_freq_dict,contingency_dict]
	
		return return_list

	def ratio_compiler(dict, list): 
		# compiles ratios for collexeme attracted/repelled
		positive_count=0
		negative_count=0

		for items in list:
			if items in dict:
				if dict[items][1] == "Attracted": positive_count+=1
				if dict[items][1] == "Infinity": positive_count+=1
				if dict[items][1] == "Repelled": negative_count+=1
				if dict[items][1] == "Neg_Infinity": negative_count+=1
		return safe_divide(positive_count,negative_count)

	def ratio_compiler_type(dict, list): 
		# compiles ratios for collexeme attracted/repelled but each type is only counted
		# once
		positive_count=0
		negative_count=0

		for items in set(list):
			if items in dict:
				if dict[items][1] == "Attracted": positive_count+=1
				if dict[items][1] == "Infinity": positive_count+=1
				if dict[items][1] == "Repelled": negative_count+=1
				if dict[items][1] == "Neg_Infinity": negative_count+=1
		return safe_divide(positive_count,negative_count)

	def band_counter(text_list, band_list, denominator):
		counter = 0
		for items in text_list:
			##print items		
			if items in band_list:
				counter+=1
		index = safe_divide(counter, denominator)
	
		return index

	def simple_database_counter(dict, list, number, stdev = "no", log = "no"):
		var_list = []
		counter=0
		n_counter=0
		for items in list:
			variable = items.split("\t")
			key = variable[number]
			if key in dict:
				if log == "yes":
					counter+= math.log10(float(dict[key]))
				else:
					counter+= float(dict[key])
				if stdev == "yes":
					var_list.append(float(dict[key]))
				n_counter+=1
		if stdev == "no":
			return safe_divide(counter,n_counter)
		if stdev == "yes":
			return var_list

	def simple_database_counter_type(dict, list, number):
		counter=0
		n_counter=0
		for items in set(list):
			variable = items.split("\t")
			key = variable[number]
			if key in dict:
				counter+= float(dict[key])
				n_counter+=1
		return safe_divide(counter,n_counter)

	def simple_database_counter_log(dict, list, number):
		counter=0
		n_counter=0
		for items in list:
			variable = items.split("\t")
			key = variable[number]
			if key in dict:
				##print float(dict[key])
				counter+= math.log10(float(dict[key]))
				n_counter+=1
		return safe_divide(counter,n_counter)

	def contingency_database_counter(dict, list, number, stdev = "no", log = "no"):
		var_list = []
		counter=0
		n_counter=0
		for items in list:
			#variable = items[0]+"\t"+items[1]
			if items in dict:
				if log == "yes":
					counter+= math.log10(float(dict[items][number]))
				else:
					counter+= float(dict[items][number])
				if stdev == "yes":
					var_list.append(float(dict[items][number]))
				n_counter+=1
		if stdev == "no":
			return safe_divide(counter,n_counter)
		if stdev == "yes":
			return var_list

	def contingency_database_counter_type(dict, list, number):
		counter=0
		n_counter=0
		for items in set(list):
			#variable = items[0]+"\t"+items[1]
			if items in dict:
				counter+= float(dict[items][number])
				n_counter+=1
		return safe_divide(counter,n_counter)

	def contingency_database_counter_log(dict, list, number):
		counter=0
		n_counter=0
		for items in list:
			#variable = items[0]+"\t"+items[1]
			if items in dict:
				##print float(dict[items][number])
				counter+= math.log10(float(dict[items][number]))
				n_counter+=1
		return safe_divide(counter,n_counter)

	def ttr(list, number, dict):
		set_list=[]
		nattest = 0
		nnattest=0
		for items in list:
			items=items.split("\t")
			variable = items[number]
			if variable in dict:
				set_list.append(variable)
				nattest+=1
			else: nnattest+=1

		outvar = [safe_divide(len(set(set_list)),len(set_list)), safe_divide(nattest,(nattest+nnattest))]
		return outvar

	def contingency_ttr(list, dict):
		set_list=[]
		nattest = 0
		nnattest=0
		for items in list:

			if items in dict:
				nattest+=1
				set_list.append(items)
			else: nnattest+=1
		outvar = [safe_divide(len(set(set_list)),len(set_list)), safe_divide(nattest,(nattest+nnattest))]
		return outvar

	def component_dicter(folder): #takes a folder of component files, turns it into dict
		dict = {}
		file_list = glob.glob(folder)
		for files in file_list:
			list = []
			comp = file(files,"rU").read().split("\n")
			for line in comp:
				line = line.encode('utf-8')
				list.append(line.split("\t"))
			if system == "M" or system == "L":
				key = files.split("/")[-1]
			if system == "W":
				key = files.split("\\")[-1]
			dict[key]=list
		
		return dict

	def freq_database_compiler(output_file, lem_list, vac_list, vc_list, min_freq_value):
		# Start Part 3:
		dataQueue.put("Starting frequency list compiler...")
		root.update_idletasks()

		try:
			from scipy import stats
			scipy_test = True
			indices = """lemma	construction	lemma-construction	LF	CF	LFC	LF_per_mil	CF_per_mil	LFC_per_mil	Direction	collexeme	collexeme_approx	faith_c_outcome	faith_v_outcome	delta_p_c_outcome	delta_p_v_outcome\n"""
		except ImportError:
			scipy_test = False
			indices = """lemma	construction	lemma-construction	LF	CF	LFC	LF_per_mil	CF_per_mil	LFC_per_mil	collexeme_approx	faith_c_outcome	faith_v_outcome	delta_p_c_outcome	delta_p_v_outcome\n"""
			
		#indices = """lemma	construction	lemma-construction	LF	CF	LFC	Direction	collexeme	collexeme_approx	faith_c_outcome	faith_v_outcome	delta_p_c_outcome	delta_p_v_outcome\n"""
		output_file.write(indices)
		n_constructions = len(vc_list)

		dataQueue.put("Building frequency lists...")
		root.update_idletasks()
		
		lemma_freq_dict = Counter(lem_list)
		vac_freq_dict = Counter(vac_list)
		verb_vac_freq_dict = Counter(vc_list)
		db = verb_vac_freq_dict.most_common()

		dataQueue.put("Calculating strength of assocation measures...")
		root.update_idletasks()
	
		
		counter = 0
		nlines = len(db)
		for lines in db:
			lemma = lines[0].split("\t")[0]
			vac = lines[0].split("\t")[1]
			verb_vac = lines[0]
			verb_vac_print = lemma +"_"+vac
			
			counter += 1
			if int((counter/nlines)*100)%10 == 0:
				update_message = "Processed " + str(int((counter/nlines)*100)) + "% of constructions in corpus"
				dataQueue.put(update_message)
			root.update_idletasks()

			#print entry_name
			LF =float(lemma_freq_dict[lemma])
			CF = float(vac_freq_dict[vac])
			LCF = float(verb_vac_freq_dict[verb_vac])
			NCNL = n_constructions - (LF + CF - LCF) #vacs that don't have the construction or the verb
			
			a = LCF #LCF frequency
			if a < min_freq_value:
				continue
			b = LF - LCF #lemma not in construction
			c = CF - LCF #construction with other lemmas
			d = NCNL
			
			sum_abcd = a+b+c+d
			expected = ((a+b)*(a+c))/sum_abcd
			
			if scipy_test == True:
				if a > expected:
					direction = "Attracted"
					pvalue = stats.fisher_exact([[a,b],[c,d]],alternative='greater')
					if pvalue[1] >= .05:
						direction = "Neutral"
						collexeme = "Neutral"
					if pvalue[1]== 0.0:
						collexeme = "Infinity"
					else:
						#print "pvalue", pvalue[1]
						collexeme = -(math.log(pvalue[1],10))

				else:
					direction = "Repelled"
					pvalue = stats.fisher_exact([[a,b],[c,d]],alternative='less')
					if pvalue[1] >= .05:
						direction = "Neutral"
						collexeme = "Neutral"
					if pvalue[1]== 0.0:
						collexeme = "Neg_Infinity"
					else: collexeme = (math.log(pvalue[1],10))
		
			faith_c_outcome = (a/(a+b))
			faith_v_outcome = (a/(a+c))
			delta_p_c_outcome = (a/(a+b)) - (c/(c+d))
			delta_p_v_outcome = (a/(a+c)) - (b/(b+d))
			collexeme_approx = delta_p_v_outcome * int(LF)

			normed_LF = (LF/n_constructions) * 1000000
			normed_CF = (CF/n_constructions) * 1000000
			normed_LCF = (LCF/n_constructions) * 1000000
			#note, outputs normed values. if non-normed desired, use LF, CF, and LCF
			if scipy_test == True:
				index_list = [verb_vac,verb_vac_print,LF,CF,LCF,normed_LF,normed_CF,normed_LCF,direction,collexeme,collexeme_approx,faith_c_outcome,faith_v_outcome,delta_p_c_outcome,delta_p_v_outcome]
			else:
				index_list = [verb_vac,verb_vac_print,LF,CF,LCF,normed_LF,normed_CF,normed_LCF,collexeme_approx,faith_c_outcome,faith_v_outcome,delta_p_c_outcome,delta_p_v_outcome]
			string_index_list = []
			for items in index_list:
				string_index_list.append(str(items))
			string = "\t".join(string_index_list)+"\n"
			output_file.write(string)


	#########################################
	
	
	inputfile = indir + "/*.txt"
	#filenames = glob.glob(inputfile)	
	
	#this section is for the database. creates database file, need to add in option:
	sca_check= check_list[0].get()
	clause_check= check_list[1].get()
	phrase_check= check_list[2].get()
	soph_check= check_list[3].get()
	component_check= check_list[4].get()
	df_check = check_list[5].get()
	xml_check = check_list[6].get()
	generate_check = check_list[7].get()
	min_freq = int(check_list[8].get()) #this is for frequency list generation
	
	print "min_freq = ", min_freq
	#outputfile = outfile
	if soph_check == 1 or phrase_check ==1 or clause_check == 1:
		outf=file(outfile, "w")

	if df_check == 1:
		if clause_check ==1 or soph_check ==1 or generate_check ==1:
			data_filename = outfile[:-4] + "_clause_database.txt"
			data_file = file(data_filename,"w")
			data_file.write("lemma\tword\tsimple_vac\tsophistication_vac\tfilled_sophistication_vac\tclause_complexity_vac\tfilename\tsentence_text\n")
		
		if phrase_check == 1:
			data_filename_2 = outfile[:-4] + "_phrase_database.txt"
			data_file_2 = file(data_filename_2,"w")
			data_file_2.write("np_type\tnp_head\tnp_type_with_modifiers\tfilled_np\tfilename\tsentence_text\n")
	
	if generate_check == 1:
		list_filename = outfile[:-4] + "_frequency_database.txt"
		freq_list_out = file(list_filename,"w")
		
	if xml_check == 1:
		mod_outdir = outfile[:-4] + "_mod_parsed/"
		#mod_outdir = "/".join(outfile.split("/")[:-1]) + "/mod_parsed/"
			
		if not os.path.exists(mod_outdir):
			os.makedirs(mod_outdir)

	outfile_vac_context = outfile[:-4]+"_vac_context.csv"		

	#######
	
	if soph_check == 1:
	
		dataQueue.put("Loading Database 1 of 5...")
		root.update_idletasks()
		acad_database_file = file(resource_path("acad_index_database_3.kdk"), "rU").read().split("\n")
		acad_lemma_freq_dict = dict_builder(acad_database_file)[0]
		acad_construction_freq_dict = dict_builder(acad_database_file)[1]
		acad_contingency_dict= dict_builder(acad_database_file)[2]

		dataQueue.put("Loading Database 2 of 5...")
		root.update_idletasks()
		news_database_file = file(resource_path("news_index_database_3.kdk"), "rU").read().split("\n")
		news_lemma_freq_dict = dict_builder(news_database_file)[0]
		news_construction_freq_dict = dict_builder(news_database_file)[1]
		news_contingency_dict= dict_builder(news_database_file)[2]

		dataQueue.put("Loading Database 3 of 5...")
		root.update_idletasks()
		mag_database_file = file(resource_path("mag_index_database_3.kdk"), "rU").read().split("\n")
		mag_lemma_freq_dict = dict_builder(mag_database_file)[0]
		mag_construction_freq_dict = dict_builder(mag_database_file)[1]
		mag_contingency_dict= dict_builder(mag_database_file)[2]

		dataQueue.put("Loading Database 4 of 5...")
		root.update_idletasks()
		fic_database_file = file(resource_path("fic_index_database_3.kdk"), "rU").read().split("\n")
		fic_lemma_freq_dict = dict_builder(fic_database_file)[0]
		fic_construction_freq_dict = dict_builder(fic_database_file)[1]
		fic_contingency_dict= dict_builder(fic_database_file)[2]

		dataQueue.put("Loading Database 5 of 5... (please be patient)")
		root.update_idletasks()
		all_database_file = file(resource_path("all_index_database_3.kdk"), "rU").read().split("\n")
		all_lemma_freq_dict = dict_builder(all_database_file)[0]
		all_construction_freq_dict = dict_builder(all_database_file)[1]
		all_contingency_dict= dict_builder(all_database_file)[2]

	if component_check == 1:
		component_dict = component_dicter(resource_path("Component*.txt"))
		component_list_dict = {}
		for comp in component_dict:
			#print comp
			for item in component_dict[comp]:
				component_list_dict[item[0]] = []
		#print component_list_dict
	#####
	

	if not os.path.exists(resource_path("parsed_files/")):
		os.makedirs(resource_path("parsed_files/"))
		
	if not os.path.exists(resource_path("to_process/")):
		os.makedirs(resource_path("to_process/"))

	if not os.path.exists(resource_path("sca_parsed_files/")):
		os.makedirs(resource_path("sca_parsed_files/"))
	
	for the_file in os.listdir(resource_path("sca_parsed_files/")):
		file_path = os.path.join(resource_path("sca_parsed_files/"), the_file)
		os.unlink(file_path)

	folder_list = [resource_path("parsed_files/"), resource_path("to_process/")]
	
	dataQueue.put("Importing corpus files (this may take a while)...")
	root.update_idletasks()	
	for folder in folder_list:
		for the_file in os.listdir(folder):
			file_path = os.path.join(folder, the_file)
			os.unlink(file_path)


		copy_files = glob.glob(indir + "/*.txt")
		###print copy_files
		for thing in copy_files:
			thing_1=thing
			if system == "M" or system == "L":
				thing = thing.split("/")[-1]
				thing = resource_path("to_process/") + thing
			elif system == "W":
				thing = thing.split("\\")[-1]
				thing = resource_path("to_process\\") + thing   
			###print "origin:",thing_1
			###print "Destination:", thing
			shutil.copyfile(thing_1, thing)
		input_folder = resource_path("to_process/")
		#input_folder = re.sub(" ", "\ ", input_folder)

		#write file_list:
		list_of_files = glob.glob(input_folder + "*.txt")
		###print "list of files ", list_of_files
		file_list_file = file(input_folder + "_filelist.txt", "w")

		file_list = (input_folder + "_filelist.txt")
		###print "file list ", file_list
		for line in list_of_files:
			line = line + "\n"	
			file_list_file.write(line)
		file_list_file.flush()
		file_list_file.close()
	
	current_directory = resource_path("")
	stan_file_list = (input_folder + "_filelist.txt")
	###print "file list ", file_list
	###print input_folder

	stan_output_folder = resource_path("parsed_files/")
	memory = "3"
	nthreads = "2"
	dataQueue.put("Starting Stanford CoreNLP...")
	root.update_idletasks()
	
	if phrase_check == 1 or clause_check == 1 or soph_check == 1 or generate_check == 1:
		#call_stan_corenlp(current_directory,stan_file_list, stan_output_folder, memory, nthreads)
		ktk.call_stan_corenlp_pos(current_directory, stan_file_list, stan_output_folder, memory, nthreads, system, dataQueue,root, parse_type = ",depparse")

		##### For Parsed File Analysis #####

		try:
			import xml.etree.cElementTree as ET
		except ImportError:
			import xml.etree.ElementTree as ET
		p_files_list = glob.glob(resource_path("parsed_files/*.xml")) # Create a list of all files in target folder
		#print p_files_list
		
		total_nfiles = len(p_files_list)


		nfiles = 1 #This is a counter to see how many files have been processed
		counter = 0

		total_nsubj_deps_list = []
		total_nsubj_pass_deps_list = []
		total_agents_deps_list = []
		total_dobj_deps_list = []
		total_pobj_deps_list = []
		total_iobj_deps_list = []
		total_ncomp_deps_list = []
	
		comp_file_list = []
	
		verb_tags = "VB VBZ VBP VBD VBN VBG".split(" ") #This is a list of verb tags
		exclusions = "aux auxpass nsubj dobj iobj amod"
		contractions = "'s 're 'm".split(" ")

		noun_tags = "NN NNS NNP NNPS VBG".split(" ") #note that VBG is included because this list is only used for looking at dependents that will be a nominal
		single_quotes = "u\u2018 u\u2019 u'\u02BC'".split(" ")

		nominals = "NN NNP NNPS NNS PRP PRP$ CD DT".split(" ")
		adjectives = "JJ JJR JJS".split(" ")
		verbs = "VB VBZ VBP VBD VBN VBG".split(" ")
		other = "RB ".split(" ")
		noun_mod = ["amod", "appos", "det", "goeswith", "mwe", "nn", "num","poss", "cop", "advmod", "advcl", "rcmod","vmod"] #note: cop is thrown in for convenience; #advmod and advcl added in .8.5 , "advmod", "advcl"

		corpus_lemma_list = []
		corpus_vac_list = []
		corpus_verb_vac_list = []
		
		for files in p_files_list: #iterate through files
			header_list = []
			index_list = []

			counter +=1
			nwords = 0
			nsent = 0
		
			punctuation = ". , ? ! ) ( % / - _ -LRB- -RRB- SYM ".split(" ")
			
			processed_update = "TAASSC has processed: " + str(nfiles) + " of " + str(total_nfiles) + " files."
			dataQueue.put(processed_update) #output for user
			root.update_idletasks()
		
			outfilename = files.split("/")[-1]
			outfilename = outfilename.replace(".xml","")
		
			comp_file_list.append(outfilename)
		
			#print outfilename
		
			tree = ET.ElementTree(file=files) #The file is opened by the XML parser
	
	
		### For Phrase Complexity ###
			all_nominal_deps = []
			all_nominal_deps_NN = []
	
			all_nominals = []
			all_nsubj = []
			all_nsubj_pass =[]
			all_agents = []
			all_dobj = []
			all_pobj = []
			all_iobj = []
			all_ncomps =[]
	
			all_nominals_NN = []
			all_nsubj_NN = []
			all_nsubj_pass_NN = []
			all_agents_NN = []
			all_dobj_NN = []
			all_pobj_NN = []
			all_iobj_NN = []
			all_ncomps_NN = []
	
			nsubj_deps_list = []
			nsubj_pass_deps_list = []
			agents_deps_list = []
			dobj_deps_list = []
			pobj_deps_list = []
			iobj_deps_list = []
			ncomp_deps_list = []

			nsubj_deps_list_NN = []
			nsubj_pass_deps_list_NN = []
			agents_deps_list_NN = []
			dobj_deps_list_NN = []
			pobj_deps_list_NN = []
			iobj_deps_list_NN = []
			ncomp_deps_list_NN = []

			all_nominals_stdev_list = []
			nsubj_stdev_list = []
			nsubj_pass_stdev_list = []
			agents_stdev_list = []
			dobj_stdev_list = []
			pobj_stdev_list = []
			iobj_stdev_list = []
			ncomp_stdev_list = []
		
			all_nominals_stdev_list_NN = []
			nsubj_stdev_list_NN = []
			nsubj_pass_stdev_list_NN = []
			agents_stdev_list_NN = []
			dobj_stdev_list_NN = []
			pobj_stdev_list_NN = []
			iobj_stdev_list_NN = []
			ncomp_stdev_list_NN = []
		
			all_PP = []
			nominal_PP = []
			nominal_PP_NN = []
			phrase_constructicon = []
		###for Phrase Complexity

		###for Clause Complexity and Sophistication

			constructicon = [] #holder for the context-free VACs
			prep_constructicon = []
			verb_constructicon = [] #holder for the verb-form sensitive VACs
			lemma_constructicon = [] #holder for the lemma-sensitive VACs
			lemma_constructicon_no_vcop = [] #holder for non-copular constructions
			lemma_constructicon_aux = []#
			prep_lemma_contructicon = []#
			constructicon_database = []#

		#### THE NEXT SECTION CONVERTS THE TREE TO AN APPROXIMATION OF -makeCopulaHead #####
			for sentences in tree.iter("sentence"):
				nsent +=1
				phrase_sentence = []

				noun_list = []
				pronoun_list = []
				for tokens in sentences.iter("token"):
					phrase_sentence.append(tokens[0].text)
					if tokens[4].text in punctuation:
						continue
					#if tokens[4].text is not "punct":
					nwords += 1
					if tokens[4].text in noun_tags:
						noun_list.append(tokens.get("id"))
					if tokens[4].text == "PRP":
						pronoun_list.append(tokens.get("id"))
			
				cop_list = [] #list of copular dependency relationships in sentences (tuples)
				cop_list_simple = []
						
				for deps in sentences.iter("dependencies"): #iterates through dependencies
										
					if deps.get('type') == "collapsed-ccprocessed-dependencies": #only iterate through cc-processed-dependencies

						for dependencies in deps.iter("dep"): # iterate through the dependencies

							if dependencies.get("type") == "cop": #if the type is copular...

								cop_list.append((dependencies[0].get("idx"),dependencies[0].text,dependencies[1].get("idx"),dependencies[1].text)) #this stores the governor idx and the governor text, and then the dep idx and dep text as a tuple
								cop_list_simple.append(dependencies[1].get("idx")) #this should be the id for the copular_verb
						
					else: sentences.remove(deps) # this does not get rid of collapsed dependencies. Not sure why.

				
				for entries in cop_list:
					##print entries
					comp_check = "no"
					for dependencies in deps.iter("dep"):
						if dependencies.get("type") == "cop" and dependencies[0].get("idx") == entries[0]: #if the dependency is copular and the item is the one we are concerned with in this iteration:
							for word in sentences.iter("token"): #iterate through tokens to find the pos tag
								##print word[0].text
								if word.get("id") == entries[0]:
									pos = word[4].text
									#nom_comp_position = word.get("id")
									##print pos

							if pos in nominals: #set appropriate relationship (this may be problematic for finite versus non-finite complements)
								dependencies.set("type", "ncomp")
								comp_check = "yes"
							if pos in adjectives:
								dependencies.set("type", "acomp")
								comp_check = "yes"
							if pos in verbs:
								dependencies.set("type", "vcomp")
							if pos in other:
								dependencies.set("type", "other")

							dependencies[0].set("idx", entries[2]) #set the governor as the cop verb
							dependencies[0].text = entries[3] #set the governor as the cop verb
							dependencies[1].set("idx", entries[0]) #set the dependent as the complement
							dependencies[1].text = entries[1] #set the dependent as the complement

							continue # if this fixed the comp, continue to the next dependency

						if dependencies.get("type") not in noun_mod: #if the dependency isn't one that would only work for an nominal (this may need tweaking):

							if dependencies.get("type") != "tmod" and comp_check == "yes":
								continue
						
							if dependencies[0].get("idx") == entries[0]: #if the governor is the previous cop governor - change to cop
								dependencies[0].set("idx", entries[2]) #changes idx
								dependencies[0].text = entries[3]	#changes text

							if dependencies[1].get("idx") == entries[0]: # if the dependent is the previous cop governor - change to cop
								dependencies[1].set("idx", entries[2]) #changes idx
								dependencies[1].text = entries[3] #changes text
							
		### END COPULA CONVERSION SECTION ###
	
		### Begin Clause Complexity Section ###
				if clause_check == 1 or soph_check == 1 or generate_check ==1: #Clause Complexity
			
					token_store = [] # This will be a holder of tuples for id, word, lemma, pos
					sentence = [] # stores all of the words so sentence can be stored
					#pos = [] #stores POS information so that it can be easily retrieved later
					verbs = []
					excluded_verbs = []
					gerunds = []
					infinitives = []
					main_verbs = []
		
					if len(list(sentences.iter("token"))) > 100:
						##print files, " Sentence_skipped"
						continue

					for tokens in sentences.iter("token"):
						token_store.append((tokens.get("id"),tokens[0].text.lower(), tokens[1].text, tokens[4].text))
						##print token_store
						sentence.append(tokens[0].text) #this is word
						#pos.append(tokens[4].text) #this is POS
						#if tokens[4].text in verb_tags:
						#	verbs.append(tokens.get("id"))
					##print verbs
					##print token_store
		
					inf = "no"
					for items in token_store:
						if items[3] in verb_tags:
							for dependents in sentences.iter("dependencies"):
								if dependents.get("type") == "collapsed-ccprocessed-dependencies":
									for dep in dependents.iter("dep"):
										if dep[1].get("idx") == items[0] and dep.get("type") in exclusions:
											excluded_verbs.append(items[0]) #adds id to excluded verbs
						if items[3] == "VBG":
							gerunds.append(items[0]) #adds id to gerunds (actually any -ing verb)
			
						if items[3] == "VB" and inf =="yes":
							infinitives.append(items[0])
			
						if items[3] == "TO":
							inf = "yes"
						else: 
							inf = "no"	
			
					for items in token_store:
						if items[0] in excluded_verbs:
							#print "excluded verb", items[0]
							continue
						if items[3] in verb_tags:
							main_verbs.append(items)
				
					#print main_verbs
					#print token_store
					for items in main_verbs:
						##print "Main_Verb: ", items
						###Differentiates between copular verbs and non-copular verbs###
						#print "cop list: ", cop_list_simple
						if items[0] in cop_list_simple:
							verb_type = "vcop"
						else: 
							verb_type = "v" 
			
						###cleans up some errors that were causing program to crash:##
						verb_form = items[2]
						word_entry = items[1].lower()

						if u'\xa0' in verb_form:
							verb_form = verb_form.replace(u'\xa0',' ')
						for apostrophe in single_quotes:
							if apostrophe in verb_form:
								verb_form = verb_form.replace(apostrophe,"'")
						if "-" in verb_form:
							verb_form = verb_form.replace("-","_")


						VAC = [[int(items[0]), verb_type, verb_form]] #format ID, v or v_cop, lemma form

						for dependencies in sentences.iter("dependencies"):
							if dependencies.get("type") == "collapsed-ccprocessed-dependencies":
								for dependents in dependencies.iter("dep"):
									if dependents[0].get("idx") == items[0]:

										dependent_type = dependents.get("type") #this allows the program to fix the copula error - nominal complements are now called "ncomp"
										dependent_id = int(dependents[1].get("idx"))
										dependent_form = dependents[1].text
							
										if dependent_type == "punct":
											continue
							
										if dependent_type == "xcomp" and token_store[(int(dependents[1].get("idx"))-1)][3] in nominals:
											dependent_type = "ncomp"

										if dependent_type == "aux" and token_store[(int(dependents[1].get("idx"))-1)][3] == "MD":
											dependent_type = "modal"

										VAC.append([dependent_id,dependent_type, dependent_form])
			
						#print infinitives
						VAC = sorted(VAC, key = lambda x:int(x[0]))
						auxilliaries = ["aux", "auxpass", "modal"]
						pre_simple_VAC = []
						simple_VAC = []
						complex_VAC = []
						prep_VAC = []
						simple_VAC_aux = []
					
						#print VAC
					
						for item in VAC:
							simple_VAC_aux.append(item[1])
							if item[1] not in auxilliaries:
								pre_simple_VAC.append(item)
			
						#print len(pre_simple_VAC), pre_simple_VAC
			
						if len(pre_simple_VAC) < 2 and str(pre_simple_VAC[0][0]) in gerunds:
							#print "g skip"
							continue

						if len(pre_simple_VAC) < 2 and str(pre_simple_VAC[0][0]) in infinitives:
							#print "skip"
							continue
			
						if len(pre_simple_VAC) < 2 and pre_simple_VAC[0][2] == "be":
							#print "be skip"
							continue
			
						for item in pre_simple_VAC:
							simple_VAC.append(item[1])
							complex_VAC.append("_".join([item[1],item[2]]))
							if "prep" in item[1] and "prepc" not in item[1]:
								prep_VAC.append("prep")
							else:
								prep_VAC.append(item[1])
						####
						simple_VAC_string = "-".join(simple_VAC).lower()
					
						#print simple_VAC_string
					
						complex_VAC_string = "-".join(complex_VAC).lower()
						#print complex_VAC_string
						if "-v_be-" in complex_VAC_string:
							complex_VAC_string = complex_VAC_string.replace("-v_be-", "-vcop_be-")
							simple_VAC_string = simple_VAC_string.replace("-v-","-vcop-")
							#print complex_VAC_string
					
						lemm_entry = verb_form.lower() + "\t" + simple_VAC_string
						lemm_entry_aux = verb_form.lower() + "\t" + "-".join(simple_VAC_aux).lower()
						lemm_prep_entry = verb_form.lower() + "\t" + "-".join(prep_VAC).lower()
					
						prep_constructicon.append("-".join(prep_VAC).lower())
						constructicon.append("-".join(simple_VAC).lower())
						#lemm_entry = lemm_entry.lower()
						lemma_constructicon.append(lemm_entry)
						if "vcop" not in lemm_entry:
							lemma_constructicon_no_vcop.append(lemm_entry) # for contingency counts
						lemma_constructicon_aux.append(lemm_entry_aux)
						prep_lemma_contructicon.append(lemm_prep_entry)
					
						if df_check ==1:
							sentence_string = " ".join(sentence)
							database_string = verb_form.lower() +"\t" + word_entry +"\t" + simple_VAC_string + "\t" + complex_VAC_string + "\t" +"-".join(simple_VAC_aux).lower()+"\t"+ outfilename + "\t" + sentence_string +"\n"
							database_string_clean = filter(lambda x: x in string.printable, database_string)
							data_file.write(database_string_clean)
						
						#for frequency databaser:
						if generate_check == 1:
							corpus_lemma_list.append(verb_form.lower())
							corpus_vac_list.append(simple_VAC_string)
							corpus_verb_vac_list.append(lemm_entry)
							
				###Clause Complexity Section ###
			
		
			##### Phrase Complexity #######
				if phrase_check == 1: #if the Phrase Complexity Box is Checked
				
					nsubj_list = []
					nsubj_pass_list = []
					agents_list = []
					dobj_list = []
					pobj_list = []
					iobj_list = []
					ncomp_list = []
				
					all_nominals_stdev_dict = {}
					nsubj_stdev_dict = {}
					nsubj_pass_stdev_dict = {}
					agents_stdev_dict = {}
					dobj_stdev_dict = {}
					pobj_stdev_dict = {}
					iobj_stdev_dict = {}
					ncomp_stdev_dict = {}

					all_nominals_stdev_dict_NN = {}
					nsubj_stdev_dict_NN = {}
					nsubj_pass_stdev_dict_NN = {}
					agents_stdev_dict_NN = {}
					dobj_stdev_dict_NN = {}
					pobj_stdev_dict_NN = {}
					iobj_stdev_dict_NN = {}
					ncomp_stdev_dict_NN = {}
				
					### This is the dictionary for the examples: ##
					nominals_const_dict = {}

				
					for dependencies in sentences.iter("dependencies"):
						if dependencies.get("type") == "collapsed-ccprocessed-dependencies":
							for dependents in dependencies.iter("dep"):
								#if dependents.get("type") == "xcomp" and dependents[1].get("idx") in noun_list:
	
								if dependents.get("type") == "nsubj":
									##print "dependent :", dependents.get("type"), " word: ", dependents[1].text
									nsubj_list.append(dependents[1].get("idx")) #add idx of nsubj to list for later retrieval
									nsubj_stdev_dict[dependents[1].get("idx")] = 0 #create dict for standard deviation
									nominals_const_dict[dependents[1].get("idx")] = [[dependents[1].get("idx"),dependents.get("type"), dependents[1].text]] #create dict for construction example
									all_nominals_stdev_dict[dependents[1].get("idx")] = 0
									all_nominals.append(dependents[1].text)
									all_nsubj.append(dependents[1].text)
									if dependents[1].get("idx") not in pronoun_list:
										nsubj_stdev_dict_NN[dependents[1].get("idx")] = 0
										all_nominals_stdev_dict_NN[dependents[1].get("idx")] = 0
										all_nominals_NN.append(dependents[1].text)
										all_nsubj_NN.append(dependents[1].text)						
					
								if dependents.get("type") == "nsubjpass":
									##print "dependent :", dependents.get("type"), " word: ", dependents[1].text
									nsubj_pass_list.append(dependents[1].get("idx"))
									nsubj_pass_stdev_dict[dependents[1].get("idx")] = 0
									nominals_const_dict[dependents[1].get("idx")] = [[dependents[1].get("idx"),dependents.get("type"), dependents[1].text]] #create dict for construction example
									all_nominals_stdev_dict[dependents[1].get("idx")] = 0
									all_nominals.append(dependents[1].text)
									all_nsubj_pass.append(dependents[1].text)
									if dependents[1].get("idx") not in pronoun_list:
										nsubj_pass_stdev_dict_NN[dependents[1].get("idx")] = 0
										all_nominals_stdev_dict_NN[dependents[1].get("idx")] = 0
										all_nsubj_pass_NN.append(dependents[1].get("idx"))
										all_nominals_NN.append(dependents[1].text)

								if dependents.get("type") == "agent":
									##print "dependent :", dependents.get("type"), " word: ", dependents[1].text
									agents_list.append(dependents[1].get("idx"))
									agents_stdev_dict[dependents[1].get("idx")] = 0
									nominals_const_dict[dependents[1].get("idx")] = [[dependents[1].get("idx"),dependents.get("type"), dependents[1].text]] #create dict for construction example
									all_nominals_stdev_dict[dependents[1].get("idx")] = 0
									all_nominals.append(dependents[1].text)
									all_agents.append(dependents[1].text)
									if dependents[1].get("idx") not in pronoun_list:
										agents_stdev_dict_NN[dependents[1].get("idx")] = 0
										all_nominals_stdev_dict_NN[dependents[1].get("idx")] = 0
										all_agents_NN.append(dependents[1].get("idx"))
										all_nominals_NN.append(dependents[1].text)
												
								if dependents.get("type") == "dobj":
									##print "dependent :", dependents.get("type"), " word: ", dependents[1].text
									dobj_list.append(dependents[1].get("idx"))
									dobj_stdev_dict[dependents[1].get("idx")] = 0
									nominals_const_dict[dependents[1].get("idx")] = [[dependents[1].get("idx"),dependents.get("type"), dependents[1].text]] #create dict for construction example
									all_nominals_stdev_dict[dependents[1].get("idx")] = 0
									all_nominals.append(dependents[1].text)
									all_dobj.append(dependents[1].text)
									if dependents[1].get("idx") not in pronoun_list:
										dobj_stdev_dict_NN[dependents[1].get("idx")] = 0
										all_nominals_stdev_dict_NN[dependents[1].get("idx")] = 0
										all_nominals_NN.append(dependents[1].text)
										all_dobj_NN.append(dependents[1].text)

								if dependents.get("type") == "pobj" or "prep" in dependents.get("type"):
									##print "dependent :", dependents.get("type"), " word: ", dependents[1].text
									pobj_list.append(dependents[1].get("idx"))
									pobj_stdev_dict[dependents[1].get("idx")] = 0
									nominals_const_dict[dependents[1].get("idx")] = [[dependents[1].get("idx"),"pobj", dependents[1].text]] #create dict for construction example
									all_nominals_stdev_dict[dependents[1].get("idx")] = 0
									all_nominals.append(dependents[1].text)#check this. Problem here?
									all_pobj.append(dependents[1].text)#check this. Problem here?
									if dependents[1].get("idx") not in pronoun_list:
										pobj_stdev_dict_NN[dependents[1].get("idx")] = 0
										all_nominals_stdev_dict_NN[dependents[1].get("idx")] = 0
										all_nominals_NN.append(dependents[1].text)
										all_pobj_NN.append(dependents[1].text)


								if dependents.get("type") == "iobj":
									##print "dependent :", dependents.get("type"), " word: ", dependents[1].text
									iobj_list.append(dependents[1].get("idx"))
									iobj_stdev_dict[dependents[1].get("idx")] = 0
									nominals_const_dict[dependents[1].get("idx")] = [[dependents[1].get("idx"),dependents.get("type"), dependents[1].text]] #create dict for construction example
									all_nominals_stdev_dict[dependents[1].get("idx")] = 0
									all_nominals.append(dependents[1].text)
									all_iobj.append(dependents[1].text)
									if dependents[1].get("idx") not in pronoun_list:
										iobj_stdev_dict_NN[dependents[1].get("idx")] = 0
										all_nominals_stdev_dict_NN[dependents[1].get("idx")] = 0
										all_nominals_NN.append(dependents[1].text)
										all_iobj_NN.append(dependents[1].text)

								if dependents.get("type") == "ncomp":
									##print "dependent :", dependents.get("type"), " word: ", dependents[1].text
									ncomp_list.append(dependents[1].get("idx"))
									ncomp_stdev_dict[dependents[1].get("idx")] = 0
									nominals_const_dict[dependents[1].get("idx")] = [[dependents[1].get("idx"),dependents.get("type"), dependents[1].text]] #create dict for construction example
									all_nominals_stdev_dict[dependents[1].get("idx")] = 0
									all_nominals.append(dependents[1].text)
									all_ncomps.append(dependents[1].text)
									if dependents[1].get("idx") not in pronoun_list:
										ncomp_stdev_dict_NN[dependents[1].get("idx")] = 0
										all_nominals_stdev_dict_NN[dependents[1].get("idx")] = 0
										all_nominals_NN.append(dependents[1].text)
										all_ncomps_NN.append(dependents[1].text)

								if "prep" in dependents.get("type"):
									all_PP.append(dependents[1].text)
									##print "prep"
					
					for dependencies in sentences.iter("dependencies"):
						if dependencies.get("type") == "collapsed-ccprocessed-dependencies":
							for dependents in dependencies.iter("dep"):
								if "prep" in dependents.get("type"):
									type_dependent = "prep"
								else: type_dependent = dependents.get("type")
								##print "type dependent: ", type_dependent
								if type_dependent == "punct":
									continue
						
								gov_id = dependents[0].get("idx")
								dep_id = dependents[1].get("idx")
								dep_text = dependents[1].text
							
								if gov_id  in nsubj_list:
									#if type_dependent == "punct":
										#print "Something is Fed with punct"
									nsubj_deps_list.append(type_dependent)
									nsubj_stdev_dict[gov_id] +=1
									nominals_const_dict[gov_id].append([dep_id,type_dependent,dep_text]) #add id, dep type, word
									all_nominals_stdev_dict[gov_id] +=1
									all_nominal_deps.append(type_dependent)
									##print "nsubj: ", dependents.get("type")

								if gov_id  in nsubj_pass_list:
									nsubj_pass_deps_list.append(type_dependent)
									nsubj_pass_stdev_dict[gov_id] +=1
									nominals_const_dict[gov_id].append([dep_id,type_dependent,dep_text]) #add id, dep type, word
									all_nominals_stdev_dict[gov_id] +=1
									all_nominal_deps.append(type_dependent)
									##print "nsubj_pass: ", dependents.get("type")	

								if gov_id  in agents_list:
									agents_deps_list.append(type_dependent)
									agents_stdev_dict[gov_id] +=1
									nominals_const_dict[gov_id].append([dep_id,type_dependent,dep_text]) #add id, dep type, word
									all_nominals_stdev_dict[gov_id] +=1
									all_nominal_deps.append(type_dependent)
									##print "nsubj_pass: ", dependents.get("type")						
					
								if gov_id  in dobj_list:
									dobj_deps_list.append(type_dependent)
									dobj_stdev_dict[gov_id] +=1
									nominals_const_dict[gov_id].append([dep_id,type_dependent,dep_text]) #add id, dep type, word
									all_nominals_stdev_dict[gov_id] +=1
									all_nominal_deps.append(type_dependent)
									##print "dobj: ", dependents.get("type")
									#if type_dependent == "advmod": #print "advmod in Sentence ", dependents[0].text, nsent
						
								if gov_id  in pobj_list:
									pobj_deps_list.append(type_dependent)
									pobj_stdev_dict[gov_id] +=1
									nominals_const_dict[gov_id].append([dep_id,type_dependent,dep_text]) #add id, dep type, word
									all_nominals_stdev_dict[gov_id] +=1
									all_nominal_deps.append(type_dependent)
									##print "pobj: ", dependents.get("type")

								if gov_id  in iobj_list:
									iobj_deps_list.append(type_dependent)
									iobj_stdev_dict[gov_id] +=1
									nominals_const_dict[gov_id].append([dep_id,type_dependent,dep_text]) #add id, dep type, word
									all_nominals_stdev_dict[gov_id] +=1
									all_nominal_deps.append(type_dependent)
									##print "iobj: ", dependents.get("type")
						
								if gov_id  in ncomp_list:
									ncomp_deps_list.append(type_dependent)
									ncomp_stdev_dict[gov_id] +=1
									nominals_const_dict[gov_id].append([dep_id,type_dependent,dep_text]) #add id, dep type, word
									all_nominals_stdev_dict[gov_id] +=1
									all_nominal_deps.append(type_dependent)
									##print "ncomp: ", dependents.get("type")
					
					
								#####Only noun tags: this excludes cardinal numbers and pronouns#####
								if gov_id  in nsubj_list and gov_id in noun_list:
									nsubj_deps_list_NN.append(type_dependent)
									nsubj_stdev_dict_NN[gov_id] +=1
									all_nominals_stdev_dict_NN[gov_id] +=1
									all_nominal_deps_NN.append(type_dependent)
									##print "nsubj: ", dependents.get("type")

								if gov_id  in nsubj_pass_list and gov_id in noun_list:
									nsubj_pass_deps_list_NN.append(type_dependent)
									nsubj_pass_stdev_dict_NN[gov_id] +=1
									all_nominals_stdev_dict_NN[gov_id] +=1
									all_nominal_deps_NN.append(type_dependent)
									##print "nsubj_pass: ", dependents.get("type")							

								if gov_id  in agents_list and gov_id in noun_list:
									agents_deps_list_NN.append(type_dependent)
									agents_stdev_dict_NN[gov_id] +=1
									all_nominals_stdev_dict_NN[gov_id] +=1
									all_nominal_deps_NN.append(type_dependent)
									##print "nsubj_pass: ", dependents.get("type")
											
								if gov_id  in dobj_list and gov_id in noun_list:
									dobj_deps_list_NN.append(type_dependent)
									dobj_stdev_dict_NN[gov_id] +=1
									all_nominals_stdev_dict_NN[gov_id] +=1
									all_nominal_deps_NN.append(type_dependent)
									##print "dobj: ", dependents.get("type")
	
								if gov_id  in pobj_list and gov_id in noun_list:
									pobj_deps_list_NN.append(type_dependent)
									pobj_stdev_dict_NN[gov_id] +=1
									all_nominals_stdev_dict_NN[gov_id] +=1
									all_nominal_deps_NN.append(type_dependent)
									##print "pobj: ", dependents.get("type")

								if gov_id  in iobj_list and gov_id in noun_list:
									iobj_deps_list_NN.append(type_dependent)
									iobj_stdev_dict_NN[gov_id] +=1
									all_nominals_stdev_dict_NN[gov_id] +=1
									all_nominal_deps_NN.append(type_dependent)
									##print "iobj: ", dependents.get("type")
						
								if gov_id  in ncomp_list and gov_id in noun_list:
									ncomp_deps_list_NN.append(type_dependent)
									ncomp_stdev_dict_NN[gov_id] +=1
									all_nominals_stdev_dict_NN[gov_id] +=1
									all_nominal_deps_NN.append(type_dependent)
									##print "ncomp: ", dependents.get("type")
				
					if df_check ==1:
						for items in nominals_const_dict:
							head = nominals_const_dict[items][0][1]
							head_word = nominals_const_dict[items][0][2]
							clean_const = []
							filled_const = []
							sorted_list = sorted(nominals_const_dict[items], key = lambda x:int(x[0]))
							for mods in sorted_list:
								clean_const.append(mods[1])
								filled_const.append(mods[1]+"_"+ mods[2])
							phrase_sentence_string = u" ".join(phrase_sentence)
							phrase_const_string = (u"\t").join([head,head_word,"-".join(clean_const), "-".join(filled_const),outfilename,phrase_sentence_string])
					
							data_file_2.write(phrase_const_string.encode('utf8') + "\n")
				
					for items in all_nominals_stdev_dict:
						all_nominals_stdev_list.append(all_nominals_stdev_dict[items])
					for items in nsubj_stdev_dict:
						nsubj_stdev_list.append(nsubj_stdev_dict[items])
					for items in nsubj_pass_stdev_dict:
						nsubj_pass_stdev_list.append(nsubj_pass_stdev_dict[items])
					for items in agents_stdev_dict:
						agents_stdev_list.append(agents_stdev_dict[items])
					for items in dobj_stdev_dict:
						dobj_stdev_list.append(dobj_stdev_dict[items])
					for items in pobj_stdev_dict:
						pobj_stdev_list.append(pobj_stdev_dict[items])
					for items in iobj_stdev_dict:
						iobj_stdev_list.append(iobj_stdev_dict[items])
					for items in ncomp_stdev_dict:
						ncomp_stdev_list.append(ncomp_stdev_dict[items])				
				
					for items in all_nominals_stdev_dict_NN:
						all_nominals_stdev_list_NN.append(all_nominals_stdev_dict_NN[items])
					for items in nsubj_stdev_dict_NN:
						nsubj_stdev_list_NN.append(nsubj_stdev_dict_NN[items])
					for items in nsubj_pass_stdev_dict_NN:
						nsubj_pass_stdev_list_NN.append(nsubj_pass_stdev_dict_NN[items])
					for items in agents_stdev_dict_NN:
						agents_stdev_list_NN.append(agents_stdev_dict_NN[items])
					for items in dobj_stdev_dict_NN:
						dobj_stdev_list_NN.append(dobj_stdev_dict_NN[items])
					for items in pobj_stdev_dict_NN:
						pobj_stdev_list_NN.append(pobj_stdev_dict_NN[items])
					for items in iobj_stdev_dict_NN:
						iobj_stdev_list_NN.append(iobj_stdev_dict_NN[items])
					for items in ncomp_stdev_dict_NN:
						ncomp_stdev_list_NN.append(ncomp_stdev_dict_NN[items])
		
			if phrase_check == 1: #if the Phrase Complexity Box is Checked
			
				#thus begins the actual calculation of Phrase-Level Indices:
				n_nsubj_deps = len(nsubj_deps_list)
				n_nsubj_pass_deps = len (nsubj_pass_deps_list)
				n_agents_deps = len(agents_deps_list)
				n_dobj_deps = len(dobj_deps_list)
				n_pobj_deps = len(pobj_deps_list)
				n_iobj_deps = len(iobj_deps_list)
				n_ncomp_deps= len(ncomp_deps_list)	
	
				n_nsubj_deps_NN = len(nsubj_deps_list_NN)
				n_nsubj_pass_deps_NN = len(nsubj_pass_deps_list_NN)
				n_agents_deps_NN = len(agents_deps_list_NN)
				n_dobj_deps_NN = len(dobj_deps_list_NN)
				n_pobj_deps_NN = len(pobj_deps_list_NN)
				n_iobj_deps_NN = len(iobj_deps_list_NN)
				n_ncomp_deps_NN = len(ncomp_deps_list_NN)

				n_all_nominal_deps = len(all_nominal_deps)
				n_all_nominal_deps_NN = len(all_nominal_deps_NN)
	
				n_all_nominals = len(all_nominals)
				n_all_nsubj = len(all_nsubj)
				n_all_nsubj_pass = len(all_nsubj_pass)
				n_all_agents = len(all_agents)
				n_all_dobj = len(all_dobj)
				n_all_pobj = len(all_pobj)
				n_all_iobj = len(all_iobj)
				n_all_ncomps = len(all_ncomps)

				n_all_nominals_NN = len(all_nominals_NN)
				n_all_nsubj_NN = len(all_nsubj_NN)
				n_all_nsubj_pass_NN = len(all_nsubj_pass_NN)
				n_all_agents_NN = len(all_agents_NN)
				n_all_dobj_NN = len(all_dobj_NN)
				n_all_pobj_NN = len(all_pobj_NN)
				n_all_iobj_NN = len(all_iobj_NN)
				n_all_ncomps_NN = len(all_ncomps_NN)
	
			###### Dictionaries for Specific Counts ######
	
				all_nominal_deps_dict = Counter(all_nominal_deps)
				nsubj_deps_dict = Counter(nsubj_deps_list)
				nsubj_pass_deps_dict = Counter(nsubj_pass_deps_list)
				agents_deps_dict = Counter(agents_deps_list)
				dobj_deps_dict = Counter(dobj_deps_list)
				pobj_deps_dict = Counter(pobj_deps_list)
				iobj_deps_dict = Counter(iobj_deps_list)
				ncomp_deps_dict = Counter(ncomp_deps_list)	
	
				all_nominal_deps_NN_dict = Counter(all_nominal_deps_NN)
				nsubj_deps_NN_dict = Counter(nsubj_deps_list_NN)
				nsubj_pass_deps_NN_dict = Counter(nsubj_pass_deps_list_NN)
				agents_deps_NN_dict = Counter(agents_deps_list_NN)
				dobj_deps_NN_dict = Counter(dobj_deps_list_NN)
				pobj_deps_NN_dict = Counter(pobj_deps_list_NN)
				iobj_deps_NN_dict = Counter(iobj_deps_list_NN)
				ncomp_deps_NN_dict = Counter(ncomp_deps_list_NN)

			#########################################	
	
				#ktk.indexer(len(all_PP),"n_all_PP",index_list,header_list)
				#ktk.indexer(all_nominal_deps.count("prep"),"n_nominal_PP",index_list,header_list)
				#ktk.indexer(all_nominal_deps_NN.count("prep"),"n_nominal_PP_NN",index_list,header_list)
				n_all_PP = len(all_PP)
				n_nominal_PP = all_nominal_deps.count("prep")
				n_nominal_PP_NN = all_nominal_deps_NN.count("prep")

			##########################################################

			##### Begin Larger Grain Index Counts #####			
				ktk.indexer(safe_divide(n_all_nominal_deps,n_all_nominals),"av_nominal_deps",index_list,header_list)
				ktk.indexer(safe_divide(n_nsubj_deps,n_all_nsubj),"av_nsubj_deps",index_list,header_list)
				ktk.indexer(safe_divide(n_nsubj_pass_deps,n_all_nsubj_pass),"av_nsubj_pass_deps",index_list,header_list)
				ktk.indexer(safe_divide(n_agents_deps,n_all_agents),"av_agents_deps",index_list,header_list)
				ktk.indexer(safe_divide(n_dobj_deps,n_all_dobj),"av_dobj_deps",index_list,header_list)
				ktk.indexer(safe_divide(n_pobj_deps,n_all_pobj),"av_pobj_deps",index_list,header_list)
				ktk.indexer(safe_divide(n_iobj_deps,n_all_iobj),"av_iobj_deps",index_list,header_list)
				ktk.indexer(safe_divide(n_ncomp_deps,n_all_ncomps),"av_ncomp_deps",index_list,header_list)
			
				ktk.indexer(safe_divide(n_all_nominal_deps_NN,n_all_nominals_NN),"av_nominal_deps_NN",index_list,header_list)
				ktk.indexer(safe_divide(n_nsubj_deps_NN,n_all_nsubj_NN),"av_nsubj_deps_NN",index_list,header_list)
				ktk.indexer(safe_divide(n_nsubj_pass_deps_NN,n_all_nsubj_pass_NN),"av_nsubj_pass_deps_NN",index_list,header_list)
				ktk.indexer(safe_divide(n_agents_deps_NN,n_all_agents_NN),"av_agents_deps_NN",index_list,header_list)
				ktk.indexer(safe_divide(n_dobj_deps_NN,n_all_dobj_NN),"av_dobj_deps_NN",index_list,header_list)
				ktk.indexer(safe_divide(n_pobj_deps_NN,n_all_pobj_NN),"av_pobj_deps_NN",index_list,header_list)
				ktk.indexer(safe_divide(n_iobj_deps_NN,n_all_iobj_NN),"av_iobj_deps_NN",index_list,header_list)
				ktk.indexer(safe_divide(n_ncomp_deps_NN,n_all_ncomps_NN),"av_ncomp_deps_NN",index_list,header_list)
			
				ktk.std_dev_calc_simple(all_nominals_stdev_list,"nominal_deps_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(nsubj_stdev_list,"nsubj_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(nsubj_pass_stdev_list,"nsubj_pass_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(agents_stdev_list,"agents_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(dobj_stdev_list,"dobj_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(pobj_stdev_list,"pobj_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(iobj_stdev_list,"iobj_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(ncomp_stdev_list,"ncomp_stdev",index_list,header_list)
			
				ktk.std_dev_calc_simple(all_nominals_stdev_list_NN,"nominal_deps_NN_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(nsubj_stdev_list_NN,"nsubj_NN_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(nsubj_pass_stdev_list_NN,"nsubj_pass_NN_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(agents_stdev_list_NN,"agents_NN_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(dobj_stdev_list_NN,"dobj_NN_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(pobj_stdev_list_NN,"pobj_NN_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(iobj_stdev_list_NN,"iobj_NN_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(ncomp_stdev_list_NN,"ncomp_NN_stdev",index_list,header_list)
			

			##### Begin Medium-Grained Index Counts ######
	
				ktk.indexer(dict_counter(all_nominal_deps_dict, "det", n_all_nominals),"det_all_nominal_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(all_nominal_deps_dict, "amod", n_all_nominals),"amod_all_nominal_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(all_nominal_deps_dict, "prep", n_all_nominals),"prep_all_nominal_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(all_nominal_deps_dict, "poss", n_all_nominals),"poss_all_nominal_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(all_nominal_deps_dict, "vmod", n_all_nominals),"vmod_all_nominal_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(all_nominal_deps_dict, "nn", n_all_nominals),"nn_all_nominal_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(all_nominal_deps_dict, "rcmod", n_all_nominals),"rcmod_all_nominal_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(all_nominal_deps_dict, "advmod", n_all_nominals),"advmod_all_nominal_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(all_nominal_deps_dict, "conj_and", n_all_nominals),"conj_and_all_nominal_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(all_nominal_deps_dict, "conj_or", n_all_nominals),"conj_or_all_nominal_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(all_nominal_deps_NN_dict, "det", n_all_nominals_NN),"det_all_nominal_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(all_nominal_deps_NN_dict, "amod", n_all_nominals_NN),"amod_all_nominal_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(all_nominal_deps_NN_dict, "prep", n_all_nominals_NN),"prep_all_nominal_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(all_nominal_deps_NN_dict, "poss", n_all_nominals_NN),"poss_all_nominal_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(all_nominal_deps_NN_dict, "vmod", n_all_nominals_NN),"vmod_all_nominal_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(all_nominal_deps_NN_dict, "nn", n_all_nominals_NN),"nn_all_nominal_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(all_nominal_deps_NN_dict, "rcmod", n_all_nominals_NN),"rcmod_all_nominal_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(all_nominal_deps_NN_dict, "advmod", n_all_nominals_NN),"advmod_all_nominal_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(all_nominal_deps_NN_dict, "conj_and", n_all_nominals_NN),"conj_and_all_nominal_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(all_nominal_deps_NN_dict, "conj_or", n_all_nominals_NN),"conj_or_all_nominal_deps_NN_struct",index_list,header_list)

			################# Begin Fine-Grained Analyses ########################

				ktk.indexer(dict_counter(nsubj_deps_dict, "det", n_all_nsubj),"det_nsubj_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(nsubj_deps_dict, "amod", n_all_nsubj),"amod_nsubj_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(nsubj_deps_dict, "prep", n_all_nsubj),"prep_nsubj_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(nsubj_deps_dict, "poss", n_all_nsubj),"poss_nsubj_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(nsubj_deps_dict, "vmod", n_all_nsubj),"vmod_nsubj_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(nsubj_deps_dict, "nn", n_all_nsubj),"nn_nsubj_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(nsubj_deps_dict, "rcmod", n_all_nsubj),"rcmod_nsubj_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(nsubj_deps_dict, "advmod", n_all_nsubj),"advmod_nsubj_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(nsubj_deps_dict, "conj_and", n_all_nsubj),"conj_and_nsubj_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(nsubj_deps_dict, "conj_or", n_all_nsubj),"conj_or_nsubj_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(nsubj_deps_dict, "det", n_all_nsubj_NN),"det_nsubj_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(nsubj_deps_dict, "amod", n_all_nsubj_NN),"amod_nsubj_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(nsubj_deps_dict, "prep", n_all_nsubj_NN),"prep_nsubj_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(nsubj_deps_dict, "poss", n_all_nsubj_NN),"poss_nsubj_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(nsubj_deps_dict, "vmod", n_all_nsubj_NN),"vmod_nsubj_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(nsubj_deps_dict, "nn", n_all_nsubj_NN),"nn_nsubj_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(nsubj_deps_dict, "rcmod", n_all_nsubj_NN),"rcmod_nsubj_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(nsubj_deps_dict, "advmod", n_all_nsubj_NN),"advmod_nsubj_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(nsubj_deps_dict, "conj_and", n_all_nsubj_NN),"conj_and_nsubj_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(nsubj_deps_dict, "conj_or", n_all_nsubj_NN),"conj_or_nsubj_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(dobj_deps_dict, "det", n_all_dobj),"det_dobj_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(dobj_deps_dict, "amod", n_all_dobj),"amod_dobj_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(dobj_deps_dict, "prep", n_all_dobj),"prep_dobj_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(dobj_deps_dict, "poss", n_all_dobj),"poss_dobj_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(dobj_deps_dict, "vmod", n_all_dobj),"vmod_dobj_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(dobj_deps_dict, "nn", n_all_dobj),"nn_dobj_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(dobj_deps_dict, "rcmod", n_all_dobj),"rcmod_dobj_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(dobj_deps_dict, "advmod", n_all_dobj),"advmod_dobj_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(dobj_deps_dict, "conj_and", n_all_dobj),"conj_and_dobj_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(dobj_deps_dict, "conj_or", n_all_dobj),"conj_or_dobj_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(dobj_deps_dict, "det", n_all_dobj_NN),"det_dobj_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(dobj_deps_dict, "amod", n_all_dobj_NN),"amod_dobj_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(dobj_deps_dict, "prep", n_all_dobj_NN),"prep_dobj_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(dobj_deps_dict, "poss", n_all_dobj_NN),"poss_dobj_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(dobj_deps_dict, "vmod", n_all_dobj_NN),"vmod_dobj_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(dobj_deps_dict, "nn", n_all_dobj_NN),"nn_dobj_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(dobj_deps_dict, "rcmod", n_all_dobj_NN),"rcmod_dobj_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(dobj_deps_dict, "advmod", n_all_dobj_NN),"advmod_dobj_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(dobj_deps_dict, "conj_and", n_all_dobj_NN),"conj_and_dobj_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(dobj_deps_dict, "conj_or", n_all_dobj_NN),"conj_or_dobj_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(pobj_deps_dict, "det", n_all_pobj),"det_pobj_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(pobj_deps_dict, "amod", n_all_pobj),"amod_pobj_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(pobj_deps_dict, "prep", n_all_pobj),"prep_pobj_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(pobj_deps_dict, "poss", n_all_pobj),"poss_pobj_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(pobj_deps_dict, "vmod", n_all_pobj),"vmod_pobj_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(pobj_deps_dict, "nn", n_all_pobj),"nn_pobj_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(pobj_deps_dict, "rcmod", n_all_pobj),"rcmod_pobj_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(pobj_deps_dict, "advmod", n_all_pobj),"advmod_pobj_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(pobj_deps_dict, "conj_and", n_all_pobj),"conj_and_pobj_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(pobj_deps_dict, "conj_or", n_all_pobj),"conj_or_pobj_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(pobj_deps_dict, "det", n_all_pobj_NN),"det_pobj_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(pobj_deps_dict, "amod", n_all_pobj_NN),"amod_pobj_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(pobj_deps_dict, "prep", n_all_pobj_NN),"prep_pobj_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(pobj_deps_dict, "poss", n_all_pobj_NN),"poss_pobj_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(pobj_deps_dict, "vmod", n_all_pobj_NN),"vmod_pobj_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(pobj_deps_dict, "nn", n_all_pobj_NN),"nn_pobj_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(pobj_deps_dict, "rcmod", n_all_pobj_NN),"rcmod_pobj_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(pobj_deps_dict, "advmod", n_all_pobj_NN),"advmod_pobj_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(pobj_deps_dict, "conj_and", n_all_pobj_NN),"conj_and_pobj_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(pobj_deps_dict, "conj_or", n_all_pobj_NN),"conj_or_pobj_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(iobj_deps_dict, "det", n_all_iobj),"det_iobj_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(iobj_deps_dict, "amod", n_all_iobj),"amod_iobj_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(iobj_deps_dict, "prep", n_all_iobj),"prep_iobj_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(iobj_deps_dict, "poss", n_all_iobj),"poss_iobj_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(iobj_deps_dict, "vmod", n_all_iobj),"vmod_iobj_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(iobj_deps_dict, "nn", n_all_iobj),"nn_iobj_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(iobj_deps_dict, "rcmod", n_all_iobj),"rcmod_iobj_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(iobj_deps_dict, "advmod", n_all_iobj),"advmod_iobj_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(iobj_deps_dict, "conj_and", n_all_iobj),"conj_and_iobj_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(iobj_deps_dict, "conj_or", n_all_iobj),"conj_or_iobj_deps_struct",index_list,header_list)
				ktk.indexer(dict_counter(iobj_deps_dict, "det", n_all_iobj_NN),"det_iobj_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(iobj_deps_dict, "amod", n_all_iobj_NN),"amod_iobj_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(iobj_deps_dict, "prep", n_all_iobj_NN),"prep_iobj_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(iobj_deps_dict, "poss", n_all_iobj_NN),"poss_iobj_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(iobj_deps_dict, "vmod", n_all_iobj_NN),"vmod_iobj_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(iobj_deps_dict, "nn", n_all_iobj_NN),"nn_iobj_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(iobj_deps_dict, "rcmod", n_all_iobj_NN),"rcmod_iobj_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(iobj_deps_dict, "advmod", n_all_iobj_NN),"advmod_iobj_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(iobj_deps_dict, "conj_and", n_all_iobj_NN),"conj_and_iobj_deps_NN_struct",index_list,header_list)
				ktk.indexer(dict_counter(iobj_deps_dict, "conj_or", n_all_iobj_NN),"conj_or_iobj_deps_NN_struct",index_list,header_list)
		
			####### Phrase Complexity ########
		
			#### Clausal Complexity ####
						#thus begins the actual calculation of indices:
			if clause_check == 1: #Clause Complexity
			
				ktk.dep_counter(lemma_constructicon, None, "cl_av_deps",index_list,header_list)
				ktk.std_dev_calc(lemma_constructicon, "cl_ndeps_std_dev",index_list,header_list)
				#ktk.ttr(prep_constructicon, "ttr_cl",index_list,header_list)
				ktk.dep_counter(lemma_constructicon, "acomp", "acomp_per_cl",index_list,header_list)
				ktk.dep_counter(lemma_constructicon, "advcl", "advcl_per_cl",index_list,header_list)
				ktk.dep_counter(lemma_constructicon, "agent", "agent_per_cl",index_list,header_list)
				ktk.dep_counter(lemma_constructicon, "cc", "cc_per_cl",index_list,header_list)
				ktk.dep_counter(lemma_constructicon, "ccomp", "ccomp_per_cl",index_list,header_list)
				ktk.dep_counter(lemma_constructicon, "conj", "conj_per_cl",index_list,header_list)
				ktk.dep_counter(lemma_constructicon, "csubj", "csubj_per_cl",index_list,header_list)
				ktk.dep_counter(lemma_constructicon, "csubjpass", "csubjpass_per_cl",index_list,header_list)
				ktk.dep_counter(lemma_constructicon, "dep","dep_per_cl",index_list,header_list)
				ktk.dep_counter(lemma_constructicon, "discourse", "discourse_per_cl",index_list,header_list)
				ktk.dep_counter(lemma_constructicon, "dobj", "dobj_per_cl",index_list,header_list)
				ktk.dep_counter(lemma_constructicon, "expl", "expl_per_cl",index_list,header_list)
				ktk.dep_counter(lemma_constructicon, "iobj", "iobj_per_cl",index_list,header_list)
				ktk.dep_counter(lemma_constructicon, "mark", "mark_per_cl",index_list,header_list)
				ktk.dep_counter(lemma_constructicon, "ncomp", "ncomp_per_cl",index_list,header_list)
				ktk.dep_counter(lemma_constructicon, "neg", "neg_per_cl",index_list,header_list)
				ktk.dep_counter(lemma_constructicon, "nsubj", "nsubj_per_cl",index_list,header_list)
				ktk.dep_counter(lemma_constructicon, "nsubjpass", "nsubjpass_per_cl",index_list,header_list)
				ktk.dep_counter(lemma_constructicon, "parataxis", "parataxis_per_cl",index_list,header_list)
				ktk.dep_counter(lemma_constructicon, "pcomp", "pcomp_per_cl",index_list,header_list)
				ktk.dep_counter_2(prep_lemma_contructicon, "prep", "prep_per_cl",index_list,header_list)
				ktk.dep_counter_2(prep_lemma_contructicon, "prepc", "prepc_per_cl",index_list,header_list) #added .8.6
				ktk.dep_counter(lemma_constructicon, "prt", "prt_per_cl",index_list,header_list)
				#ktk.dep_counter(lemma_constructicon, "ref", "ref_per_cl",index_list,header_list)
				ktk.dep_counter(lemma_constructicon, "tmod", "tmod_per_cl",index_list,header_list)
				ktk.dep_counter(lemma_constructicon, "xcomp", "xcomp_per_cl",index_list,header_list)
				ktk.dep_counter(lemma_constructicon, "xsubj", "xsubj_per_cl",index_list,header_list)
				ktk.dep_counter(lemma_constructicon, "advmod", "advmod_per_cl",index_list,header_list)	
				ktk.dep_counter(lemma_constructicon_aux, "aux", "aux_per_cl",index_list,header_list)
				ktk.dep_counter(lemma_constructicon_aux, "auxpass", "auxpass_per_cl",index_list,header_list)
				ktk.dep_counter(lemma_constructicon_aux, "modal", "modal_per_cl",index_list,header_list)
				#ktk.dep_counter(lemma_constructicon_aux, "vcop", "copula_per_cl",index_list,header_list) #tiny, meaningless correlation


			####### Syntactic Sophistication ########

			if soph_check == 1: #VAC Sophistication

				ktk.indexer(simple_database_counter(acad_lemma_freq_dict, lemma_constructicon, 0),"acad_av_lemma_freq",index_list,header_list)
				ktk.indexer(simple_database_counter(acad_construction_freq_dict, lemma_constructicon, 1),"acad_av_construction_freq",index_list,header_list)
				ktk.indexer(contingency_database_counter(acad_contingency_dict, lemma_constructicon, 0),"acad_av_lemma_construction_freq",index_list,header_list)
				ktk.indexer(contingency_database_counter(acad_contingency_dict, lemma_constructicon_no_vcop, 3),"acad_av_approx_collexeme",index_list,header_list)
				ktk.indexer(contingency_database_counter(acad_contingency_dict, lemma_constructicon_no_vcop, 4),"acad_av_faith_verb_cue",index_list,header_list)
				ktk.indexer(contingency_database_counter(acad_contingency_dict, lemma_constructicon_no_vcop, 5),"acad_av_faith_const_cue",index_list,header_list)
				ktk.indexer(contingency_database_counter(acad_contingency_dict, lemma_constructicon_no_vcop, 6),"acad_av_delta_p_verb_cue",index_list,header_list)
				ktk.indexer(contingency_database_counter(acad_contingency_dict, lemma_constructicon_no_vcop, 7),"acad_av_delta_p_const_cue",index_list,header_list)
				ktk.indexer(simple_database_counter_log(acad_lemma_freq_dict, lemma_constructicon, 0),"acad_av_lemma_freq_log",index_list,header_list)
				ktk.indexer(simple_database_counter_log(acad_construction_freq_dict, lemma_constructicon, 1),"acad_av_construction_freq_log",index_list,header_list)
				ktk.indexer(contingency_database_counter_log(acad_contingency_dict, lemma_constructicon, 0),"acad_av_lemma_construction_freq_log",index_list,header_list)
				ktk.indexer(simple_database_counter_type(acad_lemma_freq_dict, lemma_constructicon, 0),"acad_av_lemma_freq_type",index_list,header_list)
				ktk.indexer(simple_database_counter_type(acad_construction_freq_dict, lemma_constructicon, 1),"acad_av_construction_freq_type",index_list,header_list)
				ktk.indexer(contingency_database_counter_type(acad_contingency_dict, lemma_constructicon, 0),"acad_av_lemma_construction_freq_type",index_list,header_list)
				ktk.indexer(contingency_database_counter_type(acad_contingency_dict, lemma_constructicon_no_vcop, 3),"acad_av_approx_collexeme_type",index_list,header_list)
				ktk.indexer(contingency_database_counter_type(acad_contingency_dict, lemma_constructicon_no_vcop, 4),"acad_av_faith_verb_cue_type",index_list,header_list)
				ktk.indexer(contingency_database_counter_type(acad_contingency_dict, lemma_constructicon_no_vcop, 5),"acad_av_faith_const_cue_type",index_list,header_list)
				ktk.indexer(contingency_database_counter_type(acad_contingency_dict, lemma_constructicon_no_vcop, 6),"acad_av_delta_p_verb_cue_type",index_list,header_list)
				ktk.indexer(contingency_database_counter_type(acad_contingency_dict, lemma_constructicon_no_vcop, 7),"acad_av_delta_p_const_cue_type",index_list,header_list)
				ktk.indexer(ratio_compiler(acad_contingency_dict, lemma_constructicon_no_vcop),"acad_collexeme_ratio",index_list,header_list)
				ktk.indexer(ratio_compiler_type(acad_contingency_dict, lemma_constructicon_no_vcop),"acad_collexeme_ratio_type",index_list,header_list)
				ktk.indexer(ttr(lemma_constructicon, 0, acad_lemma_freq_dict)[0],"acad_lemma_ttr",index_list,header_list)
				ktk.indexer(ttr(lemma_constructicon, 1, acad_construction_freq_dict)[0],"acad_construction_ttr",index_list,header_list)
				ktk.indexer(contingency_ttr(lemma_constructicon, acad_contingency_dict)[0],"acad_lemma_construction_ttr",index_list,header_list)
				ktk.indexer(ttr(lemma_constructicon, 0, acad_lemma_freq_dict)[1],"acad_lemma_attested",index_list,header_list)
				ktk.indexer(ttr(lemma_constructicon, 1, acad_construction_freq_dict)[1],"acad_construction_attested",index_list,header_list)
				ktk.indexer(contingency_ttr(lemma_constructicon, acad_contingency_dict)[1],"acad_lemma_construction_attested",index_list,header_list)
			
				ktk.indexer(simple_database_counter(news_lemma_freq_dict, lemma_constructicon, 0),"news_av_lemma_freq",index_list,header_list)
				ktk.indexer(simple_database_counter(news_construction_freq_dict, lemma_constructicon, 1),"news_av_construction_freq",index_list,header_list)
				ktk.indexer(contingency_database_counter(news_contingency_dict, lemma_constructicon, 0),"news_av_lemma_construction_freq",index_list,header_list)
				ktk.indexer(contingency_database_counter(news_contingency_dict, lemma_constructicon_no_vcop, 3),"news_av_approx_collexeme",index_list,header_list)
				ktk.indexer(contingency_database_counter(news_contingency_dict, lemma_constructicon_no_vcop, 4),"news_av_faith_verb_cue",index_list,header_list)
				ktk.indexer(contingency_database_counter(news_contingency_dict, lemma_constructicon_no_vcop, 5),"news_av_faith_const_cue",index_list,header_list)
				ktk.indexer(contingency_database_counter(news_contingency_dict, lemma_constructicon_no_vcop, 6),"news_av_delta_p_verb_cue",index_list,header_list)
				ktk.indexer(contingency_database_counter(news_contingency_dict, lemma_constructicon_no_vcop, 7),"news_av_delta_p_const_cue",index_list,header_list)
				ktk.indexer(simple_database_counter_log(news_lemma_freq_dict, lemma_constructicon, 0),"news_av_lemma_freq_log",index_list,header_list)
				ktk.indexer(simple_database_counter_log(news_construction_freq_dict, lemma_constructicon, 1),"news_av_construction_freq_log",index_list,header_list)
				ktk.indexer(contingency_database_counter_log(news_contingency_dict, lemma_constructicon, 0),"news_av_lemma_construction_freq_log",index_list,header_list)
				ktk.indexer(simple_database_counter_type(news_lemma_freq_dict, lemma_constructicon, 0),"news_av_lemma_freq_type",index_list,header_list)
				ktk.indexer(simple_database_counter_type(news_construction_freq_dict, lemma_constructicon, 1),"news_av_construction_freq_type",index_list,header_list)
				ktk.indexer(contingency_database_counter_type(news_contingency_dict, lemma_constructicon, 0),"news_av_lemma_construction_freq_type",index_list,header_list)
				ktk.indexer(contingency_database_counter_type(news_contingency_dict, lemma_constructicon_no_vcop, 3),"news_av_approx_collexeme_type",index_list,header_list)
				ktk.indexer(contingency_database_counter_type(news_contingency_dict, lemma_constructicon_no_vcop, 4),"news_av_faith_verb_cue_type",index_list,header_list)
				ktk.indexer(contingency_database_counter_type(news_contingency_dict, lemma_constructicon_no_vcop, 5),"news_av_faith_const_cue_type",index_list,header_list)
				ktk.indexer(contingency_database_counter_type(news_contingency_dict, lemma_constructicon_no_vcop, 6),"news_av_delta_p_verb_cue_type",index_list,header_list)
				ktk.indexer(contingency_database_counter_type(news_contingency_dict, lemma_constructicon_no_vcop, 7),"news_av_delta_p_const_cue_type",index_list,header_list)
				ktk.indexer(ratio_compiler(news_contingency_dict, lemma_constructicon_no_vcop),"news_collexeme_ratio",index_list,header_list)
				ktk.indexer(ratio_compiler_type(news_contingency_dict, lemma_constructicon_no_vcop),"news_collexeme_ratio_type",index_list,header_list)
				ktk.indexer(ttr(lemma_constructicon, 0, news_lemma_freq_dict)[0],"news_lemma_ttr",index_list,header_list)
				ktk.indexer(ttr(lemma_constructicon, 1, news_construction_freq_dict)[0],"news_construction_ttr",index_list,header_list)
				ktk.indexer(contingency_ttr(lemma_constructicon, news_contingency_dict)[0],"news_lemma_construction_ttr",index_list,header_list)
				ktk.indexer(ttr(lemma_constructicon, 0, news_lemma_freq_dict)[1],"news_lemma_attested",index_list,header_list)
				ktk.indexer(ttr(lemma_constructicon, 1, news_construction_freq_dict)[1],"news_construction_attested",index_list,header_list)
				ktk.indexer(contingency_ttr(lemma_constructicon, news_contingency_dict)[1],"news_lemma_construction_attested",index_list,header_list)
			
				ktk.indexer(simple_database_counter(mag_lemma_freq_dict, lemma_constructicon, 0),"mag_av_lemma_freq",index_list,header_list)
				ktk.indexer(simple_database_counter(mag_construction_freq_dict, lemma_constructicon, 1),"mag_av_construction_freq",index_list,header_list)
				ktk.indexer(contingency_database_counter(mag_contingency_dict, lemma_constructicon, 0),"mag_av_lemma_construction_freq",index_list,header_list)
				ktk.indexer(contingency_database_counter(mag_contingency_dict, lemma_constructicon_no_vcop, 3),"mag_av_approx_collexeme",index_list,header_list)
				ktk.indexer(contingency_database_counter(mag_contingency_dict, lemma_constructicon_no_vcop, 4),"mag_av_faith_verb_cue",index_list,header_list)
				ktk.indexer(contingency_database_counter(mag_contingency_dict, lemma_constructicon_no_vcop, 5),"mag_av_faith_const_cue",index_list,header_list)
				ktk.indexer(contingency_database_counter(mag_contingency_dict, lemma_constructicon_no_vcop, 6),"mag_av_delta_p_verb_cue",index_list,header_list)
				ktk.indexer(contingency_database_counter(mag_contingency_dict, lemma_constructicon_no_vcop, 7),"mag_av_delta_p_const_cue",index_list,header_list)
				ktk.indexer(simple_database_counter_log(mag_lemma_freq_dict, lemma_constructicon, 0),"mag_av_lemma_freq_log",index_list,header_list)
				ktk.indexer(simple_database_counter_log(mag_construction_freq_dict, lemma_constructicon, 1),"mag_av_construction_freq_log",index_list,header_list)
				ktk.indexer(contingency_database_counter_log(mag_contingency_dict, lemma_constructicon, 0),"mag_av_lemma_construction_freq_log",index_list,header_list)
				ktk.indexer(simple_database_counter_type(mag_lemma_freq_dict, lemma_constructicon, 0),"mag_av_lemma_freq_type",index_list,header_list)
				ktk.indexer(simple_database_counter_type(mag_construction_freq_dict, lemma_constructicon, 1),"mag_av_construction_freq_type",index_list,header_list)
				ktk.indexer(contingency_database_counter_type(mag_contingency_dict, lemma_constructicon, 0),"mag_av_lemma_construction_freq_type",index_list,header_list)
				ktk.indexer(contingency_database_counter_type(mag_contingency_dict, lemma_constructicon_no_vcop, 3),"mag_av_approx_collexeme_type",index_list,header_list)
				ktk.indexer(contingency_database_counter_type(mag_contingency_dict, lemma_constructicon_no_vcop, 4),"mag_av_faith_verb_cue_type",index_list,header_list)
				ktk.indexer(contingency_database_counter_type(mag_contingency_dict, lemma_constructicon_no_vcop, 5),"mag_av_faith_const_cue_type",index_list,header_list)
				ktk.indexer(contingency_database_counter_type(mag_contingency_dict, lemma_constructicon_no_vcop, 6),"mag_av_delta_p_verb_cue_type",index_list,header_list)
				ktk.indexer(contingency_database_counter_type(mag_contingency_dict, lemma_constructicon_no_vcop, 7),"mag_av_delta_p_const_cue_type",index_list,header_list)
				ktk.indexer(ratio_compiler(mag_contingency_dict, lemma_constructicon_no_vcop),"mag_collexeme_ratio",index_list,header_list)
				ktk.indexer(ratio_compiler_type(mag_contingency_dict, lemma_constructicon_no_vcop),"mag_collexeme_ratio_type",index_list,header_list)
				ktk.indexer(ttr(lemma_constructicon, 0, mag_lemma_freq_dict)[0],"mag_lemma_ttr",index_list,header_list)
				ktk.indexer(ttr(lemma_constructicon, 1, mag_construction_freq_dict)[0],"mag_construction_ttr",index_list,header_list)
				ktk.indexer(contingency_ttr(lemma_constructicon, mag_contingency_dict)[0],"mag_lemma_construction_ttr",index_list,header_list)
				ktk.indexer(ttr(lemma_constructicon, 0, mag_lemma_freq_dict)[1],"mag_lemma_attested",index_list,header_list)
				ktk.indexer(ttr(lemma_constructicon, 1, mag_construction_freq_dict)[1],"mag_construction_attested",index_list,header_list)
				ktk.indexer(contingency_ttr(lemma_constructicon, mag_contingency_dict)[1],"mag_lemma_construction_attested",index_list,header_list)
			
				ktk.indexer(simple_database_counter(fic_lemma_freq_dict, lemma_constructicon, 0),"fic_av_lemma_freq",index_list,header_list)
				ktk.indexer(simple_database_counter(fic_construction_freq_dict, lemma_constructicon, 1),"fic_av_construction_freq",index_list,header_list)
				ktk.indexer(contingency_database_counter(fic_contingency_dict, lemma_constructicon, 0),"fic_av_lemma_construction_freq",index_list,header_list)
				ktk.indexer(contingency_database_counter(fic_contingency_dict, lemma_constructicon_no_vcop, 3),"fic_av_approx_collexeme",index_list,header_list)
				ktk.indexer(contingency_database_counter(fic_contingency_dict, lemma_constructicon_no_vcop, 4),"fic_av_faith_verb_cue",index_list,header_list)
				ktk.indexer(contingency_database_counter(fic_contingency_dict, lemma_constructicon_no_vcop, 5),"fic_av_faith_const_cue",index_list,header_list)
				ktk.indexer(contingency_database_counter(fic_contingency_dict, lemma_constructicon_no_vcop, 6),"fic_av_delta_p_verb_cue",index_list,header_list)
				ktk.indexer(contingency_database_counter(fic_contingency_dict, lemma_constructicon_no_vcop, 7),"fic_av_delta_p_const_cue",index_list,header_list)
				ktk.indexer(simple_database_counter_log(fic_lemma_freq_dict, lemma_constructicon, 0),"fic_av_lemma_freq_log",index_list,header_list)
				ktk.indexer(simple_database_counter_log(fic_construction_freq_dict, lemma_constructicon, 1),"fic_av_construction_freq_log",index_list,header_list)
				ktk.indexer(contingency_database_counter_log(fic_contingency_dict, lemma_constructicon, 0),"fic_av_lemma_construction_freq_log",index_list,header_list)
				ktk.indexer(simple_database_counter_type(fic_lemma_freq_dict, lemma_constructicon, 0),"fic_av_lemma_freq_type",index_list,header_list)
				ktk.indexer(simple_database_counter_type(fic_construction_freq_dict, lemma_constructicon, 1),"fic_av_construction_freq_type",index_list,header_list)
				ktk.indexer(contingency_database_counter_type(fic_contingency_dict, lemma_constructicon, 0),"fic_av_lemma_construction_freq_type",index_list,header_list)
				ktk.indexer(contingency_database_counter_type(fic_contingency_dict, lemma_constructicon_no_vcop, 3),"fic_av_approx_collexeme_type",index_list,header_list)
				ktk.indexer(contingency_database_counter_type(fic_contingency_dict, lemma_constructicon_no_vcop, 4),"fic_av_faith_verb_cue_type",index_list,header_list)
				ktk.indexer(contingency_database_counter_type(fic_contingency_dict, lemma_constructicon_no_vcop, 5),"fic_av_faith_const_cue_type",index_list,header_list)
				ktk.indexer(contingency_database_counter_type(fic_contingency_dict, lemma_constructicon_no_vcop, 6),"fic_av_delta_p_verb_cue_type",index_list,header_list)
				ktk.indexer(contingency_database_counter_type(fic_contingency_dict, lemma_constructicon_no_vcop, 7),"fic_av_delta_p_const_cue_type",index_list,header_list)
				ktk.indexer(ratio_compiler(fic_contingency_dict, lemma_constructicon_no_vcop),"fic_collexeme_ratio",index_list,header_list)
				ktk.indexer(ratio_compiler_type(fic_contingency_dict, lemma_constructicon_no_vcop),"fic_collexeme_ratio_type",index_list,header_list)
				ktk.indexer(ttr(lemma_constructicon, 0, fic_lemma_freq_dict)[0],"fic_lemma_ttr",index_list,header_list)
				ktk.indexer(ttr(lemma_constructicon, 1, fic_construction_freq_dict)[0],"fic_construction_ttr",index_list,header_list)
				ktk.indexer(contingency_ttr(lemma_constructicon, fic_contingency_dict)[0],"fic_lemma_construction_ttr",index_list,header_list)
				ktk.indexer(ttr(lemma_constructicon, 0, fic_lemma_freq_dict)[1],"fic_lemma_attested",index_list,header_list)
				ktk.indexer(ttr(lemma_constructicon, 1, fic_construction_freq_dict)[1],"fic_construction_attested",index_list,header_list)
				ktk.indexer(contingency_ttr(lemma_constructicon, fic_contingency_dict)[1],"fic_lemma_construction_attested",index_list,header_list)
			
				ktk.indexer(simple_database_counter(all_lemma_freq_dict, lemma_constructicon, 0),"all_av_lemma_freq",index_list,header_list)
				ktk.indexer(simple_database_counter(all_construction_freq_dict, lemma_constructicon, 1),"all_av_construction_freq",index_list,header_list)
				ktk.indexer(contingency_database_counter(all_contingency_dict, lemma_constructicon, 0),"all_av_lemma_construction_freq",index_list,header_list)
				ktk.indexer(contingency_database_counter(all_contingency_dict, lemma_constructicon_no_vcop, 3),"all_av_approx_collexeme",index_list,header_list)
				ktk.indexer(contingency_database_counter(all_contingency_dict, lemma_constructicon_no_vcop, 4),"all_av_faith_verb_cue",index_list,header_list)
				ktk.indexer(contingency_database_counter(all_contingency_dict, lemma_constructicon_no_vcop, 5),"all_av_faith_const_cue",index_list,header_list)
				ktk.indexer(contingency_database_counter(all_contingency_dict, lemma_constructicon_no_vcop, 6),"all_av_delta_p_verb_cue",index_list,header_list)
				ktk.indexer(contingency_database_counter(all_contingency_dict, lemma_constructicon_no_vcop, 7),"all_av_delta_p_const_cue",index_list,header_list)
				ktk.indexer(simple_database_counter_log(all_lemma_freq_dict, lemma_constructicon, 0),"all_av_lemma_freq_log",index_list,header_list)
				ktk.indexer(simple_database_counter_log(all_construction_freq_dict, lemma_constructicon, 1),"all_av_construction_freq_log",index_list,header_list)
				ktk.indexer(contingency_database_counter_log(all_contingency_dict, lemma_constructicon, 0),"all_av_lemma_construction_freq_log",index_list,header_list)
				ktk.indexer(simple_database_counter_type(all_lemma_freq_dict, lemma_constructicon, 0),"all_av_lemma_freq_type",index_list,header_list)
				ktk.indexer(simple_database_counter_type(all_construction_freq_dict, lemma_constructicon, 1),"all_av_construction_freq_type",index_list,header_list)
				ktk.indexer(contingency_database_counter_type(all_contingency_dict, lemma_constructicon, 0),"all_av_lemma_construction_freq_type",index_list,header_list)
				ktk.indexer(contingency_database_counter_type(all_contingency_dict, lemma_constructicon_no_vcop, 3),"all_av_approx_collexeme_type",index_list,header_list)
				ktk.indexer(contingency_database_counter_type(all_contingency_dict, lemma_constructicon_no_vcop, 4),"all_av_faith_verb_cue_type",index_list,header_list)
				ktk.indexer(contingency_database_counter_type(all_contingency_dict, lemma_constructicon_no_vcop, 5),"all_av_faith_const_cue_type",index_list,header_list)
				ktk.indexer(contingency_database_counter_type(all_contingency_dict, lemma_constructicon_no_vcop, 6),"all_av_delta_p_verb_cue_type",index_list,header_list)
				ktk.indexer(contingency_database_counter_type(all_contingency_dict, lemma_constructicon_no_vcop, 7),"all_av_delta_p_const_cue_type",index_list,header_list)
				ktk.indexer(ratio_compiler(all_contingency_dict, lemma_constructicon_no_vcop),"all_collexeme_ratio",index_list,header_list)
				ktk.indexer(ratio_compiler_type(all_contingency_dict, lemma_constructicon_no_vcop),"all_collexeme_ratio_type",index_list,header_list)
				ktk.indexer(ttr(lemma_constructicon, 0, all_lemma_freq_dict)[0],"all_lemma_ttr",index_list,header_list)
				ktk.indexer(ttr(lemma_constructicon, 1, all_construction_freq_dict)[0],"all_construction_ttr",index_list,header_list)
				ktk.indexer(contingency_ttr(lemma_constructicon, all_contingency_dict)[0],"all_lemma_construction_ttr",index_list,header_list)
				ktk.indexer(ttr(lemma_constructicon, 0, all_lemma_freq_dict)[1],"all_lemma_attested",index_list,header_list)
				ktk.indexer(ttr(lemma_constructicon, 1, all_construction_freq_dict)[1],"all_construction_attested",index_list,header_list)
				ktk.indexer(contingency_ttr(lemma_constructicon, all_contingency_dict)[1],"all_lemma_construction_attested",index_list,header_list)
		
			####### Syntactic Sophistication ########
		
			####### Sophistication StDev: ########
				ktk.std_dev_calc_simple(simple_database_counter(all_lemma_freq_dict, lemma_constructicon, 0,"yes"),"all_av_lemma_freq_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(simple_database_counter(all_construction_freq_dict, lemma_constructicon, 1,"yes"),"all_av_construction_freq_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(contingency_database_counter(all_contingency_dict, lemma_constructicon, 0,"yes"),"all_av_lemma_construction_freq_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(contingency_database_counter(all_contingency_dict, lemma_constructicon_no_vcop, 3,"yes"),"all_av_approx_collexeme_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(contingency_database_counter(all_contingency_dict, lemma_constructicon_no_vcop, 4,"yes"),"all_av_faith_verb_cue_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(contingency_database_counter(all_contingency_dict, lemma_constructicon_no_vcop, 5,"yes"),"all_av_faith_const_cue_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(contingency_database_counter(all_contingency_dict, lemma_constructicon_no_vcop, 6,"yes"),"all_av_delta_p_verb_cue_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(contingency_database_counter(all_contingency_dict, lemma_constructicon_no_vcop, 7,"yes"),"all_av_delta_p_const_cue_stdev",index_list,header_list)

				ktk.std_dev_calc_simple(simple_database_counter(acad_lemma_freq_dict, lemma_constructicon, 0,"yes"),"acad_av_lemma_freq_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(simple_database_counter(acad_construction_freq_dict, lemma_constructicon, 1,"yes"),"acad_av_construction_freq_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(contingency_database_counter(acad_contingency_dict, lemma_constructicon, 0,"yes"),"acad_av_lemma_construction_freq_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(contingency_database_counter(acad_contingency_dict, lemma_constructicon_no_vcop, 3,"yes"),"acad_av_approx_collexeme_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(contingency_database_counter(acad_contingency_dict, lemma_constructicon_no_vcop, 4,"yes"),"acad_av_faith_verb_cue_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(contingency_database_counter(acad_contingency_dict, lemma_constructicon_no_vcop, 5,"yes"),"acad_av_faith_const_cue_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(contingency_database_counter(acad_contingency_dict, lemma_constructicon_no_vcop, 6,"yes"),"acad_av_delta_p_verb_cue_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(contingency_database_counter(acad_contingency_dict, lemma_constructicon_no_vcop, 7,"yes"),"acad_av_delta_p_const_cue_stdev",index_list,header_list)

				ktk.std_dev_calc_simple(simple_database_counter(news_lemma_freq_dict, lemma_constructicon, 0,"yes"),"news_av_lemma_freq_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(simple_database_counter(news_construction_freq_dict, lemma_constructicon, 1,"yes"),"news_av_construction_freq_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(contingency_database_counter(news_contingency_dict, lemma_constructicon, 0,"yes"),"news_av_lemma_construction_freq_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(contingency_database_counter(news_contingency_dict, lemma_constructicon_no_vcop, 3,"yes"),"news_av_approx_collexeme_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(contingency_database_counter(news_contingency_dict, lemma_constructicon_no_vcop, 4,"yes"),"news_av_faith_verb_cue_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(contingency_database_counter(news_contingency_dict, lemma_constructicon_no_vcop, 5,"yes"),"news_av_faith_const_cue_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(contingency_database_counter(news_contingency_dict, lemma_constructicon_no_vcop, 6,"yes"),"news_av_delta_p_verb_cue_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(contingency_database_counter(news_contingency_dict, lemma_constructicon_no_vcop, 7,"yes"),"news_av_delta_p_const_cue_stdev",index_list,header_list)

				ktk.std_dev_calc_simple(simple_database_counter(mag_lemma_freq_dict, lemma_constructicon, 0,"yes"),"mag_av_lemma_freq_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(simple_database_counter(mag_construction_freq_dict, lemma_constructicon, 1,"yes"),"mag_av_construction_freq_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(contingency_database_counter(mag_contingency_dict, lemma_constructicon, 0,"yes"),"mag_av_lemma_construction_freq_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(contingency_database_counter(mag_contingency_dict, lemma_constructicon_no_vcop, 3,"yes"),"mag_av_approx_collexeme_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(contingency_database_counter(mag_contingency_dict, lemma_constructicon_no_vcop, 4,"yes"),"mag_av_faith_verb_cue_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(contingency_database_counter(mag_contingency_dict, lemma_constructicon_no_vcop, 5,"yes"),"mag_av_faith_const_cue_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(contingency_database_counter(mag_contingency_dict, lemma_constructicon_no_vcop, 6,"yes"),"mag_av_delta_p_verb_cue_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(contingency_database_counter(mag_contingency_dict, lemma_constructicon_no_vcop, 7,"yes"),"mag_av_delta_p_const_cue_stdev",index_list,header_list)

				ktk.std_dev_calc_simple(simple_database_counter(fic_lemma_freq_dict, lemma_constructicon, 0,"yes"),"fic_av_lemma_freq_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(simple_database_counter(fic_construction_freq_dict, lemma_constructicon, 1,"yes"),"fic_av_construction_freq_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(contingency_database_counter(fic_contingency_dict, lemma_constructicon, 0,"yes"),"fic_av_lemma_construction_freq_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(contingency_database_counter(fic_contingency_dict, lemma_constructicon_no_vcop, 3,"yes"),"fic_av_approx_collexeme_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(contingency_database_counter(fic_contingency_dict, lemma_constructicon_no_vcop, 4,"yes"),"fic_av_faith_verb_cue_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(contingency_database_counter(fic_contingency_dict, lemma_constructicon_no_vcop, 5,"yes"),"fic_av_faith_const_cue_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(contingency_database_counter(fic_contingency_dict, lemma_constructicon_no_vcop, 6,"yes"),"fic_av_delta_p_verb_cue_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(contingency_database_counter(fic_contingency_dict, lemma_constructicon_no_vcop, 7,"yes"),"fic_av_delta_p_const_cue_stdev",index_list,header_list)

			#####
				ktk.std_dev_calc_simple(simple_database_counter(all_lemma_freq_dict, lemma_constructicon, 0,"yes","yes"),"all_av_lemma_freq_log_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(simple_database_counter(all_construction_freq_dict, lemma_constructicon, 1,"yes","yes"),"all_av_construction_freq_log_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(contingency_database_counter(all_contingency_dict, lemma_constructicon, 0,"yes","yes"),"all_av_lemma_construction_freq_log_stdev",index_list,header_list)

				ktk.std_dev_calc_simple(simple_database_counter(acad_lemma_freq_dict, lemma_constructicon, 0,"yes","yes"),"acad_av_lemma_freq_log_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(simple_database_counter(acad_construction_freq_dict, lemma_constructicon, 1,"yes","yes"),"acad_av_construction_freq_log_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(contingency_database_counter(acad_contingency_dict, lemma_constructicon, 0,"yes","yes"),"acad_av_lemma_construction_freq_log_stdev",index_list,header_list)

				ktk.std_dev_calc_simple(simple_database_counter(news_lemma_freq_dict, lemma_constructicon, 0,"yes","yes"),"news_av_lemma_freq_log_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(simple_database_counter(news_construction_freq_dict, lemma_constructicon, 1,"yes","yes"),"news_av_construction_freq_log_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(contingency_database_counter(news_contingency_dict, lemma_constructicon, 0,"yes","yes"),"news_av_lemma_construction_freq_log_stdev",index_list,header_list)

				ktk.std_dev_calc_simple(simple_database_counter(mag_lemma_freq_dict, lemma_constructicon, 0,"yes","yes"),"mag_av_lemma_freq_log_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(simple_database_counter(mag_construction_freq_dict, lemma_constructicon, 1,"yes","yes"),"mag_av_construction_freq_log_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(contingency_database_counter(mag_contingency_dict, lemma_constructicon, 0,"yes","yes"),"mag_av_lemma_construction_freq_log_stdev",index_list,header_list)

				ktk.std_dev_calc_simple(simple_database_counter(fic_lemma_freq_dict, lemma_constructicon, 0,"yes","yes"),"fic_av_lemma_freq_log_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(simple_database_counter(fic_construction_freq_dict, lemma_constructicon, 1,"yes","yes"),"fic_av_construction_freq_log_stdev",index_list,header_list)
				ktk.std_dev_calc_simple(contingency_database_counter(fic_contingency_dict, lemma_constructicon, 0,"yes","yes"),"fic_av_lemma_construction_freq_log_stdev",index_list,header_list)


	###############		
			if component_check == 1:
				for x in range(len(header_list)):
					if header_list[x] in component_list_dict:
						component_list_dict[header_list[x]].append(index_list[x])					
					
	###############			
			if clause_check == 1 or phrase_check ==1 or soph_check ==1:
				if nfiles == 1:
					header = "filename,nwords,"+",".join(header_list) + "\n"
					outf.write(header)
				
		
				variable_string_list=[] 
				for items in index_list:
					variable_string_list.append(str(items))
				string1 = ",".join(variable_string_list)
				
				if system == "M" or system == "L":
					filename2 = files.split("/")[-1]
				if system == "W":
					filename2 = files.split("\\")[-1]
				
				
				filename2 = filename2.replace(".xml","")
		
				##print file_counter
				outf.write ('{0}, {1}, {2}\n'
				.format(filename2,nwords,string1))
		
			#output new xml_file
			if xml_check == 1:
				if system == "M" or system == "L":				
					mod_file = mod_outdir + files.split("/")[-1]
				if system == "W":
					mod_file = mod_outdir + files.split("\\")[-1]
					
				tree.write(mod_file)			

			nfiles +=1 #add to counter
		if clause_check == 1 or phrase_check ==1 or soph_check ==1:			
			outf.flush()
			outf.close()
	
	#### Creates z scores for components, then creates component scores ####
	if component_check == 1:
		comp_file = outfile[:-4] + "_components.csv"
		comp_out = file(comp_file, "w")
		comp_out.write("""filename,np_elaboration,verb_vac_frequency,nouns_as_modifiers,determiners,vac_frequency,association_strength,diversity_and_frequency,possessives,frequency\n""")
		
		def component_counter(files, dict, item,values_dict):
			list = dict[item]
			component_score_list = []
			for x in range(len(files)):
				counter = 0	
				for entries in list:
					name = entries[0]
					eigen = float(entries[1])
					number = values_dict[name][x]
					counter+= number * eigen
				component_score_list.append(str(counter))
			
			return component_score_list
				
		z_dict = {}
		for components in component_list_dict:
			#print "component: ", items
			items = component_list_dict[components]
			z_list = []
			mean = safe_divide(sum(items),len(items))
			variance = map(lambda x: (x-mean)**2, items)
			stdev = math.sqrt(safe_divide(sum(variance),len(variance)))
			for values in items:
				z_value = safe_divide(values-mean,stdev)
				z_list.append(z_value)
			z_dict[components] = z_list
		
		comp_1 = component_counter(comp_file_list, component_dict, "Component_1.txt",z_dict)
		comp_2 = component_counter(comp_file_list, component_dict, "Component_2.txt",z_dict)
		comp_3 = component_counter(comp_file_list, component_dict, "Component_3.txt",z_dict)
		comp_4 = component_counter(comp_file_list, component_dict, "Component_4.txt",z_dict)
		comp_5 = component_counter(comp_file_list, component_dict, "Component_5.txt",z_dict)
		comp_6 = component_counter(comp_file_list, component_dict, "Component_6.txt",z_dict)
		comp_7 = component_counter(comp_file_list, component_dict, "Component_7.txt",z_dict)
		comp_8 = component_counter(comp_file_list, component_dict, "Component_8.txt",z_dict)
		comp_9 = component_counter(comp_file_list, component_dict, "Component_9.txt",z_dict)
		
		for x in range(len(comp_file_list)):
			comp_out.write(",".join([comp_file_list[x],comp_1[x],comp_2[x],comp_3[x],comp_4[x],comp_5[x],comp_6[x],comp_7[x],comp_8[x],comp_9[x]])+"\n")
		comp_out.flush()
		comp_out.close()	
			
		####### Syntactic Complexity Analyzer ########
	
	if sca_check == 1: #if the SCA box is Checked
		
		if xml_check == 1:
			sca_outdir = "/".join(outfile.split("/")[:-1]) + "/sca_parsed/"
			if not os.path.exists(sca_outdir):
				os.makedirs(sca_outdir)	

		sca_file = outfile[:-4] + "_sca.csv"
		sca_out = file(sca_file, "w")

		nfiles = 1
		sca_output_folder = resource_path("sca_parsed_files/")
		
		ktk.call_stan_corenlp_pos(current_directory, stan_file_list, sca_output_folder, memory, nthreads, system, dataQueue,root, parse_type = ",parse")
				
		sca_files = glob.glob(resource_path("sca_parsed_files/*.xml"))

		
		### This is an adaptation of Xioafe Lu's Syntactic Complexity Analyzer Version 3.3.3 (updated in TAASSC 1.1 ###

		#Tregex patterns for various syntactic constructions:

		#NOTE: SEE http://www.surdeanu.info/mihai/teaching/ista555-fall13/readings/PennTreebankConstituents.html#SQ for a full list of TAGS
	
		if system == "M" or system == "L":
			treg_verb_phrase = "'VP > S|SQ|SINV'"
			treg_vp_q = "'MD|VBZ|VBP|VBD > (SQ !< VP)'" 
			treg_clause = "'S|SINV|SQ [> ROOT <, (VP <# VB) | <# MD|VBZ|VBP|VBD | < (VP [<# MD|VBP|VBZ|VBD | < CC < (VP <# MD|VBP|VBZ|VBD)])]'"
			treg_tunit = "'S|SBARQ|SINV|SQ > ROOT | [$-- S|SBARQ|SINV|SQ !>> SBAR|VP]'"
			treg_dep_clause = "'SBAR < (S|SINV|SQ [> ROOT <, (VP <# VB) | <# MD|VBZ|VBP|VBD | < (VP [<# MD|VBP|VBZ|VBD | < CC < (VP <# MD|VBP|VBZ|VBD)])])'"
			treg_complex_tunit = "'S|SBARQ|SINV|SQ [> ROOT | [$-- S|SBARQ|SINV|SQ !>> SBAR|VP]] << (SBAR < (S|SINV|SQ [> ROOT <, (VP <# VB) | <# MD|VBZ|VBP|VBD | < (VP [<# MD|VBP|VBZ|VBD | < CC < (VP <# MD|VBP|VBZ|VBD)])]))'"
			treg_coordinate_phrase = "'ADJP|ADVP|NP|VP < CC'"
			treg_complex_nominal_1 = "'NP !> NP [<< JJ|POS|PP|S|VBG | << (NP $++ NP !$+ CC)]'"
			treg_complex_nominal_2 = "'SBAR [<# WHNP | <# (IN < That|that|For|for) | <, S] & [$+ VP | > VP]'"
			treg_complex_nominal_3 = "'S < (VP <# VBG|TO) $+ VP'"
			treg_frag_clause = "'FRAG > ROOT !<< (S|SINV|SQ [> ROOT <, (VP <# VB) | <# MD|VBZ|VBP|VBD | < (VP [<# MD|VBP|VBZ|VBD | < CC < (VP <# MD|VBP|VBZ|VBD)])])'"
			treg_frag_tunit = "'FRAG > ROOT !<< (S|SBARQ|SINV|SQ > ROOT | [$-- S|SBARQ|SINV|SQ !>> SBAR|VP])'"

		if system == "W":
			treg_verb_phrase = '"VP > S|SQ|SINV"'
			treg_vp_q = '"MD|VBZ|VBP|VBD > (SQ !< VP)"' #new in TAASSC 1.1
			treg_clause = '"S|SINV|SQ [> ROOT <, (VP <# VB) | <# MD|VBZ|VBP|VBD | < (VP [<# MD|VBP|VBZ|VBD | < CC < (VP <# MD|VBP|VBZ|VBD)])]"'
			treg_tunit = '"S|SBARQ|SINV|SQ > ROOT | [$-- S|SBARQ|SINV|SQ !>> SBAR|VP]"'
			treg_dep_clause = '"SBAR < (S|SINV|SQ [> ROOT <, (VP <# VB) | <# MD|VBZ|VBP|VBD | < (VP [<# MD|VBP|VBZ|VBD | < CC < (VP <# MD|VBP|VBZ|VBD)])])"'
			treg_complex_tunit = '"S|SBARQ|SINV|SQ [> ROOT | [$-- S|SBARQ|SINV|SQ !>> SBAR|VP]] << (SBAR < (S|SINV|SQ [> ROOT <, (VP <# VB) | <# MD|VBZ|VBP|VBD | < (VP [<# MD|VBP|VBZ|VBD | < CC < (VP <# MD|VBP|VBZ|VBD)])]))"'
			treg_coordinate_phrase = '"ADJP|ADVP|NP|VP < CC"'
			treg_complex_nominal_1 = '"NP !> NP [<< JJ|POS|PP|S|VBG | << (NP $++ NP !$+ CC)]"'
			treg_complex_nominal_2 = '"SBAR [<# WHNP | <# (IN < That|that|For|for) | <, S] & [$+ VP | > VP]"'
			treg_complex_nominal_3 = '"S < (VP <# VBG|TO) $+ VP"'
			treg_frag_clause = '"FRAG > ROOT !<< (S|SINV|SQ [> ROOT <, (VP <# VB) | <# MD|VBZ|VBP|VBD | < (VP [<# MD|VBP|VBZ|VBD | < CC < (VP <# MD|VBP|VBZ|VBD)])])"'
			treg_frag_tunit = '"FRAG > ROOT !<< (S|SBARQ|SINV|SQ > ROOT | [$-- S|SBARQ|SINV|SQ !>> SBAR|VP])"'
		

		try:
			import xml.etree.cElementTree as ET
		except ImportError:
			import xml.etree.ElementTree as ET

		total_nfiles = len(sca_files)
		nfiles = 0
		
		for files in sca_files:
			index_list = []
			header_list = []
			nfiles +=1
			
			processed_update = "L2SCA has processed: " + str(nfiles) + " of " + str(total_nfiles) + " files."
			dataQueue.put(processed_update) #output for user
			root.update_idletasks()
			
			tree = ET.ElementTree(file=files)
			
			nwords = 0
			nsent = 0
			
			punctuation = ". , ? ! ) ( % / - _ -LRB- -RRB- SYM ".split(" ")
			
			for sentences in tree.iter("sentence"):
				nsent +=1
				for items in sentences.findall("dependencies"):
					sentences.remove(items)
					
				for tokens in sentences.iter("token"):
					if tokens[4].text in punctuation:
						continue
					#if tokens[4].text is not "punct":
					nwords += 1
				
			
			tregex_list = [treg_verb_phrase,treg_clause,treg_tunit,treg_dep_clause,treg_complex_tunit,treg_coordinate_phrase,treg_complex_nominal_1,treg_complex_nominal_2,treg_complex_nominal_3,treg_frag_clause,treg_frag_tunit,treg_vp_q] #changed in TAASSC 1.1
			tregex_list_count = []
			
			for item in tregex_list:
				if system == "M" or system == "L":
					tregex_call = "java -Xmx16g -cp stanford-tregex.jar: edu.stanford.nlp.trees.tregex.TregexPattern "+ item + " " + files + " -C -o"
				if system == "W":
					tregex_call = "java -mx100m -cp "+resource_path("")+"*; edu.stanford.nlp.trees.tregex.TregexPattern "+ item + " " + files + " -C -o"
				tregex_call = subprocess.Popen(tregex_call, shell=True, stdout=subprocess.PIPE)
				count = tregex_call.communicate()[0].split('\n')[0]
				tregex_list_count.append(int(count))
				##print count
	
			#number of verb phrases:	
			nverb_phrase = tregex_list_count[0] + tregex_list_count[11] #changed in TAASSC 1.1
	
			#number of clauses:
			nclauses=tregex_list_count[1]+tregex_list_count[9]
	
			#number of t-units:
			ntunits=tregex_list_count[2]+tregex_list_count[10]
	
			#number of dependent clauses:
			ndep_clause = tregex_list_count[3]
	
			#number of complex t-units:
			ncomplex_tunit = tregex_list_count[4]
	
			#number of coordinate phrases:
			ncoordinate_phrase = tregex_list_count[5]
	
			#number of complex nominals:
			ncomplex_nominal=tregex_list_count[6]+tregex_list_count[7]+tregex_list_count[8]

			ktk.indexer(safe_divide(nwords,nsent),"MLS",index_list,header_list)
			ktk.indexer(safe_divide(nwords,ntunits),"MLT",index_list,header_list)
			ktk.indexer(safe_divide(nwords,nclauses),"MLC",index_list,header_list)
			ktk.indexer(safe_divide(nclauses,nsent),"C_S",index_list,header_list)
			ktk.indexer(safe_divide(nverb_phrase,ntunits),"VP_T",index_list,header_list)
			ktk.indexer(safe_divide(nclauses,ntunits),"C_T",index_list,header_list)
			ktk.indexer(safe_divide(ndep_clause, nclauses),"DC_C",index_list,header_list)
			ktk.indexer(safe_divide(ndep_clause, ntunits),"DC_T",index_list,header_list)
			ktk.indexer(safe_divide(ntunits, nsent),"T_S",index_list,header_list)
			ktk.indexer(safe_divide(ncomplex_tunit, ntunits),"CT_T",index_list,header_list)
			ktk.indexer(safe_divide(ncoordinate_phrase,ntunits),"CP_T",index_list,header_list)
			ktk.indexer(safe_divide(ncoordinate_phrase,nclauses),"CP_C",index_list,header_list)
			ktk.indexer(safe_divide(ncomplex_nominal,ntunits),"CN_T",index_list,header_list)
			ktk.indexer(safe_divide(ncomplex_nominal,nclauses),"CN_C",index_list,header_list)

			if nfiles == 1:
				header = "filename,nwords,"+",".join(header_list) + "\n"
				sca_out.write(header)
			#nfiles +=1 #add to counter
		
			variable_string_list=[] 
			for items in index_list:
				variable_string_list.append(str(items))
			string1 = ",".join(variable_string_list)
		
			filename2 = files.split("/")[-1]
			filename2 = filename2.replace(".xml","")
		
			##print file_counter
			sca_out.write ('{0}, {1}, {2}\n'
			.format(filename2,nwords,string1))
			
			if xml_check == 1:
				if system == "M" or system == "L":
					sca_file = sca_outdir + files.split("/")[-1]
				if system == "W":
					sca_file = sca_outdir + files.split("\\")[-1]
				tree.write(sca_file)
			
					
		sca_out.flush()
		sca_out.close()

	#for frequency lister
	if generate_check ==1:
		freq_database_compiler(freq_list_out, corpus_lemma_list,corpus_vac_list,corpus_verb_vac_list,min_freq)

		
	if df_check ==1:
		if clause_check==1:
			data_file.flush()
			data_file.close()
		
		if phrase_check ==1:
			data_file_2.flush()
			data_file_2.close()
	
	if sca_check == 1:
		nfiles = len(sca_files)
	else: nfiles = len(p_files_list)
	
	######Clean-up:
	folder_list = [resource_path("parsed_files/"), resource_path("to_process/"),resource_path("sca_parsed_files/")]

	for folder in folder_list:
		if os.path.exists(folder):
			for the_file in os.listdir(folder):
				file_path = os.path.join(folder, the_file)
				os.unlink(file_path)
	#######
	
	finishmessage = ("Processed " + str(nfiles) + " Files")
	dataQueue.put(finishmessage)
	root.update_idletasks()
	if system == "M":
		#self.progress.config(text =finishmessage)
		import tkMessageBox
		tkMessageBox.showinfo("Finished!", "Your files have been processed by TAASSC")
	

class Catcher:
	def __init__(self, func, subst, widget):
		self.func = func
		self.subst = subst
		self.widget = widget

	def __call__(self, *args):
		try:
			if self.subst:
				args = apply(self.subst, args)
			return apply(self.func, args)
		except SystemExit, msg:
			raise SystemExit, msg
		except:
			import traceback
			import tkMessageBox
			ermessage = traceback.format_exc(1)
			ermessage = re.sub(r'.*(?=Error)', "", ermessage, flags=re.DOTALL)
			ermessage = "There was a problem processing your files:\n\n"+ermessage
			tkMessageBox.showerror("Error Message", ermessage)

if __name__ == '__main__':		
	root = tk.Tk()
	root.wm_title("TAASSC 1.3.8")
	root.configure(background = color)
	#sets starting size: NOTE: it doesn't appear as though Tkinter will let you make the 
	#starting size smaller than the space needed for widgets.
	root.geometry(geom_size)
	tk.CallWrapper = Catcher
	myapp = MyApp(root)
	root.mainloop()

