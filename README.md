# Pyjade

Joseph Muller

jhmuller@umich.edu

Pyjade is a program for interacting with the Omeka Classic MySQL database used by the [Jane Addams Digital Edition](https://github.com/joemull/pyjade/). The program exports, transforms, and curates datasets for use by data modelers, researchers, and programmers. Export format options include XLSX, CSV, JSON, HTML, and TXT.

## Installing the environment
The program runs on Python 3.7 and several external libraries, which are installed by the virtual environment library [`virtualenv`](https://pypi.org/project/virtualenv/).

1. Clone or download the repository and navigate into it with Terminal (Mac) or [Git Bash](https://gitforwindows.org/) (Windows).
2. If you do not have `virtualenv` installed, install it with `pip install virtualenv` or [another installation method](https://virtualenv.pypa.io/en/latest/installation.html).
3. Create a virtual environment.
    ```
    virtualenv venv
    ```
4. Activate the virtualenv.

    Mac
    ```
    source venv/bin/activate
    ```
    Windows Git Bash
    ```
    source venv/Scripts/activate
    ```

5. Install the external libraries.
    ```
    pip install -r requirements.txt
    ```
6. Deactivate the virtual environment.
    ```
    deactivate
    ```

## Running the program
1. Create a copy of `example_options.json` and name it `options.json`.
2. Activate the virtual environment if not already activated (see above).
3. Edit `options.json` as desired--see (see [Options](#Options))
4. Run the program.
    ```
    python pyjade.py
    ```
5. Choose between partial or full ingest.
    ```
    Loaded options file
     'part' to test script (by using limit=1000 on the main SQL queries)
     'full' to run full export
    pyjade: part
    ```
6. Choose between cached or fresh data
    ```
     'cached' to load from cache (if just output settings were changed)
     'fresh' to load from RDBMS
    pyjade: fresh
    ```
7. View time-stamped outputs in `outputs` directory.
8. Repeat steps 3–7 as needed.

## Options
The `options.json` is used to set up the SQL connection, control what data is exported, and provide frequently referenced information for ingesting the item elements from the SQL database.

### `SQL`
Put in parameters for your SQL connection. Visit [`mysql-connector-python`](https://github.com/mysql/mysql-connector-python) for details.

### `DATASET_OPTIONS`
* `EXPORT` Whether to export any data. Useful to set `false` if you are just trying to ingest data from SQL. Overrides the subsequent five format export options if set to `false`.
* `EXPORT_XLSX` Whether to include XLSX in each dataset.
* `EXPORT_CSV` Whether to include CSV in each dataset.
* `EXPORT_JSON` Whether to include JSON in each dataset.
* `WRAP_JSON_RECORDS_IN_TYPE_BRANCHES` Recommended `false`, so that foreign key in relations (`jade_id` in object position) can be easily looked up in one large dictionary without having to sort by type. If set to `true`, will segment the object-level dictionary into up to six separate dictionaries named by type.
* `EXPORT_TXT` Whether to include TXT in each dataset.
* `EXPORT_HTML` Whether to include HTML in each dataset.
* `EXPORT_COMPLETE_CURATED_DATASET` Whether to export a complete dataset. (The total items ingested and exported can still be capped at runtime with `part`—see above).
* `EXPORT_SUBSETS` Whether to export any subsets, as specified below.
* `SUBSET_PROPERTIES_AND_QUANTITIES` Set the property names for which to export value subsets (if `EXPORT_SUBSETS` is `true`), and how many sets to make for each. The script takes the top values by frequency of occurrence. For example,
  ```json
  "dcterms_subject" : 15,
  ```
  will export fifteen subsets containing objects tagged with the fifteen most frequently occurring subjects.
* `OTHER_SUBSET_PROPERTIES_AND_QUANTITIES` Use as a place to hold property names you don't want to export.
* `PROPERTIES_TO_INCLUDE_FOR_EACH_TYPE` Defines what properties are considered regular for each type. Include or exclude properties to export more or less data for each type.
* `EXPORT_SEPARATE_SQL_CRUMBS` Properties that occur irregularly in the RDBMS (regular as defined above) will be considered noise by the program and can be optionally exported as `sql_crumbs.xlsx`).
* `EXPORT_EVERYTHING_IN_SINGLE_XLSX_SHEET` Whether to create one big spreadsheet with all objects and properties. Includes lots of blank cells. Overwhelming but useful if you just want to see all the data in one place.
* `OUTPUT_STATS` Whether to create markdown and XLSX files with tables of property and relation counts. Tables may have bugs, so edit before publishing.

### ELEMENT_DICTIONARY
Recommended this does not change. Used by the script to translate SQL IDs to labels and know which elements to ingest (mostly `DCTERMS_IN_USE` and `DESC_JADE_ELEMENTS`) and which object types to keep (`TYPES`). Most administrative data are excluded.
