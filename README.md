# Table Statement Verification and Evidence Finding

This repository contains code for my SemEval20212 Task 9 approach within the context of my thesis project. The aim is to develop a system for table statement verification and evidence finding. The use case for this project is to promote proper interpretation of scientific articles by helping users understand the information presented in tables.

## Tasks

Two tasks are defined by the SemEval Task organizers: 

### Table Statement Support

The first subtask of this project is to determine whether a table supports a given statement. This involves finetuning a multi-modal DocFormer model for a table NLI task to analyze the table data and determine whether the statement is supported by the information presented.

### Relevant Cell Selection

The second subtask of this project is to identify the cells in the table that provide evidence for the statement. This involves developing a method to interpret the predictions of the DocFormer model.

## Data

The data used for this project consists of tables from scientific articles and corresponding verification statements. These tables are extracted from XMLs and transformed into PNGs, then augmented to randomly add the "Unknown" class to the training data. 

