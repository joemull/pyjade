'''
pyjade

A program to export, curate, and transform data from the MySQL database used by the Jane Addams Digital Edition.

'''

import os
import re
import sys
import json
import string
import datetime

import mysql.connector
from diskcache import Cache
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from tqdm import tqdm
from safeprint import print

'''
Options
'''
try:        # Options file setup credit Sam Sciolla
    with open(os.path.join('options.json')) as env_file:
        ENV = json.loads(env_file.read())
except:
    print('"Options.json" not found; please add "options.json" to the current directory.')

'''
SQL Connection
'''
DB = mysql.connector.connect(
  host=ENV['SQL']['HOST'],
  user=ENV['SQL']['USER'],
  passwd=ENV['SQL']['PASSWORD'],
  database=ENV['SQL']['DATABASE']
)
CUR = DB.cursor(buffered=True)


'''
Setup
'''
BEGIN = datetime.datetime.now()
TS = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
ITEM_ELEMENTS = ENV['ELEMENT_DICTIONARY']['DCTERMS_IN_USE']
ITEM_ELEMENTS.update(ENV['ELEMENT_DICTIONARY']['DESC_JADE_ELEMENTS'])
TYPES = ENV['ELEMENT_DICTIONARY']['TYPES']
OUT_DIR = 'outputs/'
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
DATASET_OPTIONS = ENV['DATASET_OPTIONS']
CRUMBS = DATASET_OPTIONS['EXPORT_SEPARATE_SQL_CRUMBS']
PROP_SET_LIST = DATASET_OPTIONS['PROPERTIES_TO_INCLUDE_FOR_EACH_TYPE']
INCLUDE_PROPS = DATASET_OPTIONS['PROPERTIES_TO_INCLUDE_FOR_EACH_TYPE']


class Dataset():
    def __init__(self):

        '''
        Start building the dataset objects by pulling IDs and types from omek_items
        '''

        statement = '''
            SELECT omek_items.id as item_id, omek_item_types.`name` as 'jade_type', collection_id as 'jade_collection' FROM omek_items
			JOIN omek_item_types on omek_items.item_type_id = omek_item_types.id
            WHERE public = 1
            ORDER BY item_id;
        '''
        self.omek_items = pd.read_sql(statement,DB)
        self.omek_items = self.omek_items.set_index('item_id',drop=False)
        self.objects = self.omek_items.copy()
        self.objects['item_id'] = self.objects['item_id'].apply(
            lambda x: self.convert_to_jade_id(x))
        self.objects.rename(columns={'item_id': 'jade_id'},inplace=True)
        self.objects = self.objects.set_index('jade_id',drop=False)
        self.objects = self.objects[self.objects['jade_type'].isin(
            ['Text','Event','Person','Organization','Publication']
        )]

        # Noise is an alternate dataset to record property values that dont fit the regular usage
        self.noise = self.objects.copy()
        self.noise.drop('jade_type',axis=1)
        self.noise.drop('jade_collection',axis=1)

    def ingest(self,limit=None):

        '''
        Get the item element texts
        '''

        statement = f'''
            SELECT et.id AS id, et.record_id AS record_id,
                et.element_id AS element_id, et.`text` AS el_text,
                items.item_type_id AS item_type
            FROM omek_element_texts as et
            JOIN omek_items AS items ON et.record_id = items.id
            WHERE record_type = "Item"
            ORDER BY id;
        '''
        if limit != None:
            statement = statement.split(';')[0] + f' LIMIT {str(limit)};'
        self.element_texts = pd.read_sql(statement,DB)

        # Load environment variables
        ELEMENT_IDS = list(ITEM_ELEMENTS.keys())

        # Set data structure:
        data = {}
        noise = {}

        # Iterate through the element_texts
        iter = tqdm(self.element_texts.iterrows())
        iter.set_description("Ingesting item attributes")
        for tup in iter:
            row = tup[1]
            element_id = str(row.loc['element_id'])
            if row.loc['record_id'] in self.omek_items.index.values:
                jade_type = self.omek_items.loc[row.loc['record_id'],'jade_type']
                jade_id = self.convert_to_jade_id(row.loc['record_id'])

                # Filter element texts through environment variables
                if element_id in ELEMENT_IDS:
                    if jade_type in TYPES.values():
                        element_label = ITEM_ELEMENTS[element_id]

                        # Filters property values through the sets designated in the options
                        if element_label in INCLUDE_PROPS[jade_type]:
                            compile_json(data,jade_id,element_label,row.loc['el_text'])
                        else:
                            compile_json(noise,jade_id,element_label,row.loc['el_text'])
                            # if CRUMBS:
                                # print('Excluded',element_label,'in type',jade_type)


        # Add accumulated data to DataFrame
        new_df = pd.DataFrame.from_dict(data,orient='index')
        new_noise_df = pd.DataFrame.from_dict(noise,orient='index')
        self.objects = pd.concat([self.objects,new_df],axis=1)
        self.noise = pd.concat([self.noise,new_noise_df],axis=1)

        # Add URLs
        base_url = "https://digital.janeaddams.ramapo.edu/items/show/"
        self.objects.insert(loc=1,column='jade_url',value=[
                base_url+id.split('_')[-1] for id in self.objects.index.values
            ])

        self.add_collections(limit)
        self.add_tags(limit)

        # Remove records with no title fields found
        self.objects = self.objects.dropna(subset=['dcterms_title'])

    def convert_to_jade_id(self,item_id):

        '''
        Prepend the type string to the SQL primary key so that locations and items are unique in the same set of relations
        '''

        if type(item_id) != type(str):
            if item_id in self.omek_items.index.values:
                the_type = self.omek_items.at[item_id,"jade_type"]
                if the_type in list(TYPES.values()):
                    return the_type.lower()+"_"+str(item_id)
                else:
                    return "unspecified_"+str(item_id)
            else:
                return "unpublished_"+str(item_id)
        else:
            return item_id

    def add_tags(self,limit):

        '''
        Pull tags from the database
        '''

        statement = f'''
        SELECT * FROM omek_records_tags
        JOIN omek_tags on omek_records_tags.tag_id = omek_tags.id;
        '''

        self.tag_df = pd.read_sql(statement,DB)
        self.objects = self.objects[:limit].apply(
            lambda x : self.add_tag(x),axis=1)

    def add_tag(self, row_ser):

        '''
        Add the tag to the list for each object
        '''

        new_subj_field = []
        id = row_ser.loc['jade_id']

        try:
            tag_names = self.tag_df.loc[self.tag_df['record_id'] == int(id.split("_")[-1])]
            if not tag_names.empty:
                for name in tag_names['name'].to_list():
                    if name not in new_subj_field:
                        new_subj_field.append(name)
            row_ser['dcterms_subject'] = new_subj_field
            return row_ser
        except:
            return row_ser

    def add_collections(self,limit):

        '''
        Pull collections from the database
        '''

        statement = '''
            SELECT omek_collections.id as collection_id, `text` as collection_name FROM omek_collections
    	        JOIN omek_element_texts AS texts ON omek_collections.id = texts.record_id
                    WHERE record_type = "Collection"
                    AND element_id = 50
                    AND public = 1;
        '''
        self.collection_df = pd.read_sql(statement,DB)
        self.collection_df = self.collection_df.set_index('collection_id')
        self.objects = self.objects[:limit].apply(
            lambda x : self.add_collection(x),
            axis=1
        )

    def add_collection(self,row_ser):

        '''
        Add the collection to the list for each object
        '''

        new_collection_field = []
        ids = row_ser.loc['jade_collection']
        if not isinstance(ids, list):
            ids = [ids]

        try:
            for coll_id in ids:
                matches = self.collection_df.at[coll_id,'collection_name']

                if isinstance(matches,np.ndarray):
                    match_list = matches.tolist()
                elif isinstance(matches,str):
                    match_list = [matches]
                else:
                    print("Unrecognized type of collection",type(matches))

                for name in match_list:
                    if name not in new_collection_field:
                        new_collection_field.append(name)
            row_ser['jade_collection'] = new_collection_field
            return row_ser
        except:
            return row_ser

    def add_relations(self,limit=None):

        '''
        Ingest relation data from SQL
        '''

        # Read from SQL tables omek_item_relations_relations and omek_item_relations_properties
        statement = f'''
            SELECT relations.id as id, relations.subject_item_id AS subjId, properties.id as relId, properties.label AS relLabel, relations.object_item_id AS objId
            FROM omek_item_relations_relations AS relations
            	JOIN omek_item_relations_properties AS properties ON relations.property_id = properties.id;
        '''
        if limit != None:
            statement = statement.split(';')[0] + f' LIMIT {str(limit)};'
        self.relations = pd.read_sql(statement,DB,index_col='id')

        # Style relation labels with camel case
        self.relations['relLabel'] = self.relations['relLabel'].apply(
            lambda x: camel(x))

        # Set up data structure
        data = {}
        noise = {}

        # Add the type prefix to the subject and object IDs
        self.relations['subjId'] = self.relations['subjId'].apply(
            lambda x: self.convert_to_jade_id(x))

        self.relations['objId'] = self.relations['objId'].apply(
            lambda x: self.convert_to_jade_id(x))

        # Iterate through the relation set
        iter = tqdm(self.relations.iterrows())
        iter.set_description("Adding relations")
        for tup in iter:
            row = tup[1]
            subjId = row['subjId']
            relLabel = row['relLabel']
            objId = row['objId']

            if (
                subjId in self.objects.index.values
            ) and (
                objId in self.objects.index.values
            ):
                # print(subjId,objId)
                compile_json(data,subjId,relLabel,objId)
            else:
                compile_json(noise,subjId,relLabel,objId)

        # Add locations to the relations
        # This is a thorny call bramble that should probably be untangled in a future iteration of the script
        locSet = LocationSet()
        locSet.ingest(self,limit=limit)
        data, noise = self.add_locations(locSet,data,noise)

        # Add the compiled relation data into the main DataFrame and the noise bin
        new_df = pd.DataFrame(data={"jade_relation":list(data.values())},index=list(data.keys()))
        self.objects = pd.concat([self.objects,new_df],sort=False,axis=1)

        new_noise_df = pd.DataFrame(data={"jade_relation":list(noise.values())},index=list(noise.keys()))
        self.noise = pd.concat([self.noise,new_noise_df],sort=False,axis=1)

    def add_locations(self,locSet,data,noise):

        '''
        Add locations from class object already constructed
        '''

        # Add the type prefix to the location and item IDs
        locSet.locations['loc_id'] = locSet.locations['loc_id'].astype(str)
        locSet.locations['loc_id'] = locSet.locations['loc_id'].apply(
            lambda x : "location_" + str(x))
        locSet.locations.rename(columns={'loc_id': 'jade_id'},inplace=True)

        # Merge locations table into objects table
        self.objects = pd.concat([self.objects,locSet.locations],axis=0)
        self.objects = self.objects.set_index('jade_id',drop=False)
        self.objects.index.name = None
        dataset_ids = self.objects.index.values
        self.location_duplicates = locSet.location_duplicates

        # Iterate through the location set
        iter = tqdm(locSet.locations.iterrows())
        iter.set_description("Adding locations")
        for tup in iter:
            row = tup[1]

            # Iterate through the collection of items for each location
            for rel in list(row.loc['loc_relation'].items()):
                loc_id = row.loc['jade_id']
                desc_list = rel[1]
                item_id = rel[0]

                for desc in desc_list:

                    # Build up the data structure for the later DataFrame
                    if item_id in dataset_ids:
                        compile_json(data,item_id,desc,loc_id)
                    else:
                        compile_json(noise,item_id,desc,loc_id)

        # Remove relations from locations table as they are now represented in item rows
        self.objects = self.objects.drop("loc_relation",axis=1)

        # Add location types
        self.objects = self.objects.apply(
            lambda ser : self.add_location_types(ser),
            axis=1
        )
        self.noise = self.noise.apply(
            lambda ser : self.add_location_types(ser),
            axis=1
        )

        self.objects = self.objects.dropna(subset=['jade_id'])

        return data, noise

    def add_location_types(self,row):

        '''
        Look for null type values and adds location if location in jade_id prefix
        '''

        try:
            if pd.isnull(row.loc['jade_type']):
                if type(row.loc['jade_id']) == type(""):
                    if row.loc['jade_id'].split("_")[0] == "location":
                        row.loc['jade_type'] = "Location"
                    else:
                        print("Type null but not location:",row)

                else:
                    print('Dropped type not included:',row['jade_url'])
            return row
        except:
            print("Unknown problem during adding location type for:",row)

    def quantify(self):

        '''
        Run counting functions on properties and relations to create descriptive statistics about the data
        '''

        self.quant = {}

        # Items
        self.quant["item_property_count"] = self.objects.count()

        # Item properties
        self.quantify_properties()

        # Item properties by type
        self.quantify_properties_by_type()

        # Relations (including location relations)
        self.quantify_relations()

        # Data nesting
        self.quant['nesting'] = {}
        self.check_nesting(self.objects)

    def quantify_properties(self):

        '''
        Run counts of properties
        '''

        # Iterate through properties identified for faceting
        props = list(DATASET_OPTIONS['SUBSET_PROPERTIES_AND_QUANTITIES'].items())
        iter = tqdm(props)
        iter.set_description("Quantifying subsets by facet")
        for prop, lim in iter:
            if prop in self.objects.columns.values:

                # Special cases
                if prop in ['dcterms_date']:

                    # Date
                    dc_dates_ser = self.objects[prop]
                    dc_dates_ser = dc_dates_ser.apply(unwrap_list)
                    dc_dates_ser = dc_dates_ser.dropna()
                    for id in dc_dates_ser.index.values:
                        try:
                            date_val = dc_dates_ser[id]
                            if not isinstance(date_val, list):
                                date_list = [date_val]
                            else:
                                date_list = date_val
                            for date_string in date_list:
                                if not isinstance(date_string, str):
                                    date_string = str(date_string)
                                yearlike = date_string.split('-')[0]
                                if (
                                    len(yearlike) == 4
                                ) and (
                                    int(yearlike[0]) == 1
                                ) and (
                                    yearlike[3] in '0123456789'
                                ):
                                    year = yearlike
                                    dc_dates_ser[id] = str(year)
                                else:
                                    dc_dates_ser.drop(id)
                                    print('Dropped unrecognized date value:',id,dc_dates_ser[id])
                        except:
                            dc_dates_ser.drop(id)
                            print('Dropped unrecognized date value:',id,dc_dates_ser[id])
                    if len(dc_dates_ser) > 1:
                        self.add_to_quant(
                            dc_dates_ser,
                            sort_on_property_name=False)

                # All others / standard structure
                else:
                    ser = self.objects[prop]
                    ser = ser.dropna()
                    if len(ser) > 1:
                        self.add_to_quant(ser)

    def add_to_quant(
        self,
        series,    # A named Series object whose index is the item or location IDs
                    # and whose values are non-empty strings or lists of strings
        sort_on_property_name = False # Default False sorts by largest count. Optional True sorts alphabetically by property name
        ):

        '''
        Index the DataFrame's IDs by value of passed property (column name)
        '''

        property = series.name

        # Create an index of jade_ids by property value for the series (column) passed
        for id in series.index.values:
            cell = series[id]
            if isinstance(cell, np.ndarray):
                cell = cell.tolist()
            if not isinstance(cell, list):
                cell = [cell]
            for val in cell:
                compile_json(
                    self.quant,
                    property,
                    val.strip() if isinstance(val, str) else val,
                    id)

        # Create a dictionary of property values and instance counts
        for val in list(self.quant[property].keys()):
            compile_json(self.quant,
                property+"_count",
                val,
                len(self.quant[property][val]))

        # Sort the dictionary and add it to the dataset object
        if not sort_on_property_name:
            self.quant[property+"_count"] = dict(
                sort_by_item_counts(self.quant[property+"_count"]))
        self.quant[property+"_count"] = pd.Series(
            self.quant[property+"_count"],
            index=list(self.quant[property+"_count"].keys()),
            name=property+"_count")
        if sort_on_property_name:
            self.quant[property+"_count"] = self.quant[property+"_count"].sort_index()

        # Go ahead and unwrap the single-integer lists created by compile_json
        self.quant[property+"_count"] = self.quant[property+"_count"].apply(unwrap_list)

    def quantify_properties_by_type(self):

        '''
        Create a table of property counts by object type
        '''

        # Get a copy of the main DataFrame and send each row through the counter
        self.quant['prop_table'] = {}
        df = self.objects.copy()
        df = df.apply(
            lambda ser : self.compile_types_by_prop(ser),
            axis=1
        )

        # Make the resulting dict a DataFrame, sort it, and abbreviate column headers
        self.quant['prop_table'] = pd.DataFrame.from_dict(
            self.quant['prop_table'],
            orient='index')
        self.quant['prop_table'] = self.quant['prop_table'][[
            'Person',
            'Text',
            'Event',
            'Organization',
            'Publication',
            'Location',
            'All Types'
        ]]
        self.quant['prop_table'] = self.quant['prop_table'].sort_index()
        self.quant['prop_table'].rename(columns={'Organization':'Org.', 'Publication':'Pub.', 'Location':'Loc.'},inplace=True)

    def compile_types_by_prop(self,ser):

        '''
        Count the properties in the passed series by object type
        '''

        jade_type = ser.loc['jade_type']
        jade_type = unwrap_list(jade_type)
        if jade_type in list(INCLUDE_PROPS.keys()):
            for prop in ser.index.values:
                if prop in INCLUDE_PROPS[jade_type]:
                    cell = ser.loc[prop]
                    if not isinstance(cell, list):
                        cell = [cell]
                    if not pd.isnull(cell).any():
                        if prop not in self.quant['prop_table']:
                            self.quant['prop_table'][prop] = {}

                        if "All Properties" not in self.quant['prop_table']:
                            self.quant['prop_table']['All Properties'] = {}

                        if jade_type not in self.quant['prop_table'][prop]:
                            self.quant['prop_table'][prop][jade_type] = 1
                        else:
                            self.quant['prop_table'][prop][jade_type] += 1

                        if "All Types" not in self.quant['prop_table'][prop]:
                            self.quant['prop_table'][prop]["All Types"] = 1
                        else:
                            self.quant['prop_table'][prop]["All Types"] += 1

                        if jade_type not in self.quant['prop_table']['All Properties']:
                            self.quant['prop_table']['All Properties'][jade_type] = 1
                        else:
                            self.quant['prop_table']['All Properties'][jade_type] += 1

        return ser

    def quantify_relations(self):

        '''
        Make a list of unique relation triples and a table of the most common subject–object pairs
        '''

        # Iterate through relations in the Dataset
        uniq_rels = {}
        count_df_index = []
        count_df_columns = []
        iter = tqdm(self.objects.index.values)
        iter.set_description("Counting unique relations")
        for subjId in iter:
            row = self.objects.loc[subjId]
            row_rels_dict = row.loc['jade_relation']
            if not pd.isnull(row_rels_dict):
                for relLabel, objIdList in row_rels_dict.items():
                    for objId in objIdList:

                        # Find the types of each subject and object
                        subjType = subjId.split('_')[0].capitalize()
                        objType = objId.split('_')[0].capitalize()

                        # Count the unique combinations of subject, relation, and object
                        rel = " ".join([subjType,relLabel,objType])

                        if rel not in uniq_rels:
                            uniq_rels[rel] = 1
                        else:
                            uniq_rels[rel] += 1

                        # Make the dimensions for a dataframe
                        if subjType not in count_df_index:
                            count_df_index.append(subjType)
                        if objType not in count_df_columns:
                            count_df_columns.append(objType)

        # Sort and output simple list
        self.quant["unique_relation_list"] = pd.DataFrame.from_dict(
            dict(sort_by_item_counts(uniq_rels)),orient='index')

        # Make the dataframe
        count_df = pd.DataFrame(data=0,index=count_df_index,columns=count_df_columns)
        for rel in list(uniq_rels.keys()):
            count = uniq_rels[rel]
            try:
                subjType, relLabel, objType = rel.split(' ')
                count_df.at[subjType,objType] += count
            except:
                print("Error counting relation:",rel)
        self.quant["unique_relation_table"] = count_df

    def check_nesting(self,df):

        '''
        Check whether each column in the passed df has repeating values in any of the rows
        '''

        for prop in df.columns.values:
            column_ser = df[prop]
            column_ser = column_ser.dropna()
            self.is_nested(column_ser)

    def is_nested(self,ser):

        '''
        Is the passed row repeating/nested?
        '''

        nested = False
        for id, val in ser.iteritems():
            if (
                type(val) == type([])
            ) or (
                type(val) == type({})
            ):
                if len(val) > 1:
                    nested = True
        self.quant['nesting'][ser.name] = nested

    def unwrap_nonrepeating_columns(self):

        '''
        If a column hasn't been marked as nested, take its values out of the list wrappers
        '''

        for prop in self.objects.columns.values:
            if not self.quant['nesting'][prop]:
                self.objects[prop] = self.objects[prop].apply(unwrap_list)

    def segment_by_type(self,df):

        '''
        Break up the passed dataframe by object type and return up to six separate frames that only have the properties belonging to their types
        '''

        type_segments = {}
        for type_name in list(PROP_SET_LIST.keys()):
            prospective_props = PROP_SET_LIST[type_name]
            props_for_this_type = []
            for prop in prospective_props:
                if prop in df.columns.values:
                    props_for_this_type.append(prop)
            segment_df = df[props_for_this_type]
            segment_df = segment_df.loc[lambda text_df: text_df['jade_type'] == type_name, :]
            type_segments[type_name] = segment_df

        return type_segments

    def export_stats(self):

        '''
        Export results from quantify to an XLSX file
        '''

        filepath = f'{OUT_DIR}{TS}-batch/'
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        with open(
            filepath+"jade_data_stats.md",
            'w',
            encoding='utf-8'
        ) as md_writer:
            with pd.ExcelWriter(
                filepath+"jade_data_stats.xlsx",
                encoding='utf-8'
            ) as excel_writer:
                for k in list(self.quant.keys()):
                    if k.split("_")[-1] in ["count", "list", "table"]:
                        md_writer.write(f"\n\n## {k}\n"+self.quant[k].to_markdown())
                        if isinstance(self.quant[k], pd.Series):
                            df = self.quant[k].apply(lambda x : colons_and_semicolons(x))
                            df = df.apply(lambda x: zap_illegal_characters(x))
                        else:
                            df = self.quant[k].applymap(lambda x : colons_and_semicolons(x))
                            df = df.applymap(lambda x: zap_illegal_characters(x))
                        df.to_excel(excel_writer,sheet_name=k)

    def export_single_sheet(self):

        '''
        Export one big sheet that has all the objects and all the properties and relations (contains a lot of blank cells)
        '''

        filepath = f'{OUT_DIR}{TS}-batch/'
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        with pd.ExcelWriter(
            filepath+"jade_data_single_sheet.xlsx",
            encoding='utf-8'
        ) as excel_writer:
            df = self.objects.applymap(lambda x : colons_and_semicolons(x))
            df = df.applymap(lambda x: zap_illegal_characters(x))
            df.to_excel(excel_writer,index=False,sheet_name='jade_data')

    def export_complete_dataset(self):

        '''
        Export a complete, curated dataset, segmented by object type in the XLSX and CSV formats
        '''

        self.type_segments = self.segment_by_type(self.objects)
        filepath = f'{OUT_DIR}{TS}-batch/complete_data/'
        self.run_outputs(self.type_segments,filepath)

        # filepath = f'{OUT_DIR}{TS}-batch/complete_data/Locations'
        # self.run_outputs(self.locations,filepath)

    def export_subsets(self):

        '''
        Manage creation of subsets by property value, using quant information
        '''

        props = list(DATASET_OPTIONS['SUBSET_PROPERTIES_AND_QUANTITIES'].items())
        iter = tqdm(props)
        iter.set_description("Exporting subsets by facet")
        for prop, lim in iter:

            if prop in self.quant:
                self.create_subset(
                    prop,
                    self.quant[prop],
                    self.quant[prop+'_count'],
                    lim)

    def create_subset(self,prop,attr_dict,ranked_attr_counts,lim):

        '''
        Create a subset for the passed property, using indexes in quant
        '''

        ranked_attr_list = list(ranked_attr_counts.keys())
        for val in ranked_attr_list[:lim]:
            filtered_jade_ids = attr_dict[val]
            count = str(ranked_attr_counts[val])

            # Items
            df = self.objects[self.objects.index.isin(filtered_jade_ids)]
            segmented_subset_dfs = self.segment_by_type(df)
            safe_val_string = safen_string(val)
            filepath = f'{OUT_DIR}{TS}-batch/filtered_subsets/{prop}/{safe_val_string} {count}/'
            self.run_outputs(segmented_subset_dfs,filepath,filename=f'{prop} {safe_val_string} {count}')

    def export_crumbs(self):

        '''
        Export a spreadsheet with noise from the RDBMS that did not conform to regular property usage. Does not yet contain relation noise. May have a bug with location noise, including too many locations. Also has a bug with respect to jade_id and jade_collection, leaving all of the regular values for those properties in.
        '''

        filepath = f'{OUT_DIR}{TS}-batch/'
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        with pd.ExcelWriter(
            filepath+"sql_crumbs.xlsx",
            encoding='utf-8'
        ) as excel_writer:
            df = self.noise.applymap(lambda x : colons_and_semicolons(x))
            df = df.applymap(lambda x: zap_illegal_characters(x))
            df.to_excel(excel_writer,index=False,sheet_name='item_noise')

            df = self.location_duplicates.applymap(lambda x : colons_and_semicolons(x))
            df = df.applymap(lambda x: zap_illegal_characters(x))
            df.to_excel(excel_writer,index=False,sheet_name='location_noise')

    def run_outputs(self,type_segment_dfs,filepath,filename='default'):

        '''
        Manages the outputs specified for the dfs passed
        '''

        if not os.path.exists(filepath):
            os.makedirs(filepath)

        tsdfs = type_segment_dfs
        if DATASET_OPTIONS['EXPORT_XLSX']:
            self.save_xlsx(tsdfs,filepath,filename)
        if DATASET_OPTIONS['EXPORT_CSV']:
            self.save_csv(tsdfs,filepath,filename)
        if DATASET_OPTIONS['EXPORT_JSON']:
            self.save_json(tsdfs,filepath,filename)

        text_df = tsdfs['Text']
        if (
            DATASET_OPTIONS['EXPORT_TXT']
        ) or (
            DATASET_OPTIONS['EXPORT_HTML']
        ):
            if len(text_df) > 0:
                self.save_txt_and_html(text_df,filepath,filename)

    def save_xlsx(self,tsdfs,filepath,filename):

        '''
        Run an XLSX export, putting multiple tables in a single workbook
        '''

        with pd.ExcelWriter(
            f"{filepath}{'jade_data' if filename == 'default' else filename}.xlsx",
            encoding='utf-8'
        ) as excel_writer:
            for name, df in list(tsdfs.items()):
                df = df.applymap(lambda x : colons_and_semicolons(x))
                df = df.applymap(lambda x: zap_illegal_characters(x))
                if len(df) > 0:
                    df.to_excel(excel_writer,index=False,sheet_name=name)

    def save_csv(self,tsdfs,filepath,filename):

        '''
        Run a CSV export, using a subdirectory for multiples
        '''

        filepath+=f"{'jade_data' if filename == 'default' else filename}_csv"
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        for name, df in list(tsdfs.items()):
            if len(df) > 0:
                df.to_csv(f'{filepath}/jade_{name}.csv',index=False)

    def save_json(self,tsdfs,filepath,filename):

        '''
        Run a JSON export, putting all the objects at the same level (no type segments) or wrapping them, depending on options
        '''

        json_output = {}

        if DATASET_OPTIONS['WRAP_JSON_RECORDS_IN_TYPE_BRANCHES']:
            for name, df in list(tsdfs.items()):
                json_output[name] = json.loads(df.to_json(orient='index'))

        if not DATASET_OPTIONS['WRAP_JSON_RECORDS_IN_TYPE_BRANCHES']:
            for name, df in list(tsdfs.items()):
                json_output.update(json.loads(df.to_json(orient='index')))

        with open(filepath+f"{'jade_data' if filename == 'default' else filename}.json",'w') as fileref:
            fileref.write(json.dumps(json_output))

    def save_txt_and_html(self,df,filepath,filename):

        '''
        Run export of texts, using subdirectories by format
        '''

        if DATASET_OPTIONS['EXPORT_TXT']:
            txt_filepath = filepath+f"{'jade_texts' if filename == 'default' else filename}_txt/"
            if not os.path.exists(txt_filepath):
                os.makedirs(txt_filepath)

        if DATASET_OPTIONS['EXPORT_HTML']:
            html_filepath = filepath+f"{'jade_texts' if filename == 'default' else filename}_html/"
            if not os.path.exists(html_filepath):
                os.makedirs(html_filepath)

        # Iterate through the text column
        text_ser = df["jade_text"]
        text_ser = text_ser.dropna()
        text_ser = text_ser.apply(unwrap_list)
        for jade_id, val in text_ser.iteritems():

            # Manage whether values are wrapped in lists
            if not isinstance(val, list):
                val_list = [val]
            for val in val_list:
                if not pd.isnull(val):

                    # Check whether value is html
                    is_html = False
                    if "<" in val:
                        if ">" in val:
                            is_html = True

                    # Run HTML and TXT exports
                    if is_html:
                        soup = BeautifulSoup(val,'html.parser')

                        if DATASET_OPTIONS['EXPORT_HTML']:
                            with open(html_filepath+jade_id+'.html','w',encoding='utf-8') as html_ref:
                                html_ref.write(soup.prettify())

                        if DATASET_OPTIONS['EXPORT_TXT']:
                            with open(txt_filepath+jade_id+'.txt','w',encoding='utf-8') as txt_ref:
                                txt_ref.write(text_with_newlines(soup))

                    else:
                        if DATASET_OPTIONS['EXPORT_TXT']:
                            with open(txt_filepath+jade_id+'.txt','w',encoding='utf-8') as txt_ref:
                                txt_ref.write(val)



class LocationSet():

    '''
    A class to hold locations in the few seconds before they get subsumed into the dataset object
    '''

    # A dummy init function
    def __init__(self):
        pass

    # Ingest location data from SQL
    def ingest(self,dataset,limit=None):

        # Read from SQL table omek_locations
        statement = f'''
            SELECT * FROM omek_locations;
        '''
        if limit != None:
            statement = statement.split(';')[0] + f' LIMIT {str(limit)};'
        self.omek_locations = pd.read_sql(statement,DB)

        # Set up data structure for later DataFrame
        data = {}
        noise = {}
        ids = []
        retrieved = []

        # Convert item IDs
        self.omek_locations['item_id'] = self.omek_locations['item_id'].apply(
            lambda x: dataset.convert_to_jade_id(x))

        # Read data retrieved from SQL
        iter = tqdm(self.omek_locations.iterrows())
        iter.set_description("Ingesting locations")
        for tup in iter:
            row = tup[1]
            loc_id = row.loc['id']
            if (
                loc_id not in retrieved
            ) and (
                row.loc['item_id'] in dataset.objects.index.values
            ):
                cluster_address_versions = {}

                # Check for duplicates
                addr_fp = fingerprint(row.loc["address"])
                cluster_statement = f'''
                    SELECT * FROM omek_locations
                        WHERE latitude = {row.loc['latitude']}
                        AND longitude = {row.loc['longitude']};
                '''
                cluster = pd.read_sql(cluster_statement,DB)

                # Combine duplicates
                for cluster_tup in cluster.iterrows():
                    cluster_row = cluster_tup[1]
                    if fingerprint(cluster_row.loc['address']) == addr_fp:

                        # Keep track of addresses to choose most common style below
                        if cluster_row.loc["address"] not in cluster_address_versions:
                            cluster_address_versions[cluster_row.loc["address"]] = 1
                        else:
                            cluster_address_versions[cluster_row.loc["address"]] += 1

                        # Group item-location relations, styling descriptions with camel case and defining blanks
                        cluster_loc_id = cluster_row.loc['id']
                        cluster_item_id = cluster_row.loc['item_id']
                        if (cluster_row.loc['description'] == '' or None):
                            cluster_desc = 'noDescription'

                        else:
                            cluster_desc = camel(cluster_row.loc['description'])

                        # Put approved forms in the curated data
                        compile_json(
                            data,
                            loc_id,
                            "loc_relation",
                            dataset.convert_to_jade_id(cluster_item_id),
                            cluster_desc)

                        # Keep track of which rows have been combined
                        compile_json(
                            noise,
                            loc_id,
                            "set_of_dup_loc_ids_with_assoc_item_ids",
                            cluster_loc_id,
                            cluster_item_id)

                        retrieved.append(cluster_loc_id)

                # Update address for row to most commonly used capitalization and punctuation
                chosen_style = sort_by_item_counts(cluster_address_versions)[0][0]
                data[loc_id]['jade_address'] = chosen_style
                noise[loc_id]['jade_address'] = chosen_style

                # Add in other properties
                data[loc_id]['loc_id'] = loc_id
                # data[loc_id]['jade_zoom_level'] = row.loc['zoom_level']
                # data[loc_id]['jade_map_type'] = row.loc['map_type']
                data[loc_id]['jade_latitude'] = row.loc['latitude']
                data[loc_id]['jade_longitude'] = row.loc['longitude']


        # Create DataFrame
        self.locations = pd.DataFrame.from_dict(data,orient='index')
        self.location_duplicates = pd.DataFrame.from_dict(noise,orient='index')

def fingerprint(address):

    '''
    A rudimentary string matching tool that strips everything except letters and numbers
    '''

    address_fingerprint = ''
    for l in address.lower():
        if l in string.ascii_lowercase + string.digits:
            address_fingerprint += l
    return address_fingerprint

def camel(phrase_with_spaces,cap_first=False):

    '''
    Convert to camel case
    '''

    if len(phrase_with_spaces) == 0:
        return ''
    else:
        capped_list = [w.capitalize() for w in phrase_with_spaces.split()]
        if not cap_first:
            new_list = [capped_list[0].lower()]
            new_list.extend(capped_list[1:])
            return "".join(new_list)
        else:
            return "".join(capped_list)

def compile_json(data,subj,relLabel,obj,obj2=None):

    '''
    This function nests the passed objects into a JSON tree, assuming that "data" is already an existing dictionary. If only four objects are passed, the fourth will appear in list structure. If five are passed, the fifth will be a list. The function does not return anything because dictionaries are mutable objects.
    '''

    if subj not in data:
        data[subj] = {}
    if obj2 == None:
        try:
            if relLabel not in data[subj]:
                data[subj][relLabel] = []
        except:
            print(subj,relLabel,obj)
        if obj not in data[subj][relLabel]:
            data[subj][relLabel].append(obj)
    else:
        secondRelLabel = obj
        if relLabel not in data[subj]:
            data[subj][relLabel] = {}
        if secondRelLabel not in data[subj][relLabel]:
            data[subj][relLabel][secondRelLabel] = []
        if obj2 not in data[subj][relLabel][secondRelLabel]:
            data[subj][relLabel][secondRelLabel].append(obj2)

def sort_by_item_counts(count_dict):

    '''
    Sort a dictionary by the greatest integer (value)
    '''

    return sorted(
        count_dict.items(),
        key = lambda x : x[1],
        reverse = True
    )

def unwrap_list(something):

    """
    Single things don't need to be in lists for some purposes
    """

    if (isinstance(something, list)) and (len(something) == 1):
        return something[0]
    else:
        return something

def safen_string(a_string):

    """
    Don't save folders ending with periods, for example
    """

    if not isinstance(a_string, str):
        return a_string
    safe_string = ''
    for l in a_string:
        if l in string.whitespace+string.ascii_letters+string.digits+'-':
            safe_string += l
    return safe_string.strip()

def text_with_newlines(elem):

    '''
    A custom alternative to BeautifulSoup's string methods that keeps line breaks represented by div and br elements. Whitespace and line breaks in JADE transcriptions are often used to represent spatial distance in the analog page.
    Credit: https://gist.github.com/zmwangx/ad0830ba94b1fd98f428
    '''

    text = ''
    for e in elem.descendants:
        if isinstance(e, str):
            text += e
        elif e.name == 'br' or e.name == 'p' or e.name == 'div':
            text += '\n'
    return text

def colons_and_semicolons(val):

    '''
    A manager for making lists and dictionaries more human-readable, for nested values in XLSX and CSV formats
    '''

    if isinstance(val, list):
        val = pylist_to_human_list(val)
    elif isinstance(val, dict):
        val = pydict_to_semicolon_list(val)
    return val

def pylist_to_human_list(pylist,separator=';'):

    '''
    Brackets hurt my eyes
    '''

    returnable_string = ''
    if len(pylist) >= 1:
        returnable_string += str(pylist[0]).strip()
    if len(pylist) > 1:
        for item in pylist[1:]:
            returnable_string +=f"{separator} "+str(item).strip()
    return returnable_string

def pydict_to_semicolon_list(pydict):

    '''
    Braces hurt my eyes too
    '''

    returnable_string = ''
    tup_list = list(pydict.items())
    if len(tup_list) > 0:
        first_one = True
        for tup in tup_list:
            if first_one:
                returnable_string+=str(tup[0])
            else:
                returnable_string+="; "+str(tup[0])
            returnable_string+=": "+pylist_to_human_list(tup[1],separator=',')
            first_one = False
    return returnable_string

def zap_illegal_characters(value):

    '''
    Somewhere in the JADE dataset, there are unescaped unicode characters that are in openpyxl's illegal characters list. This escapes all unicode characters in a passed string if it contains any of those illegal characters.
    Source: https://openpyxl.readthedocs.io/en/2.4/_modules/openpyxl/cell/cell.html
    '''

    if value == None:
        return
    if isinstance(value, str):
        ILLEGAL_CHARACTERS_RE = re.compile(r'[\000-\010]|[\013-\014]|[\016-\037]')
        if next(ILLEGAL_CHARACTERS_RE.finditer(value), None):
            value = value.encode('unicode_escape').decode('utf-8')
    return value

if __name__ == '__main__':

    '''
    This run sequence depends on options.json, but it also asks for two decisions
    '''

    print('Loaded options file')

    # Do you want to do just a little bit or all of it?
    limit=1000
    part_or_full = input(
        f" 'part' to test script (by using limit={str(limit)} on the main SQL queries)\n 'full' to run full export \npyjade: ")
    if part_or_full == 'full':
        limit=None

    # Are you just running the program because you changed the export options?
    cache_or_fresh = input(
        " 'cached' to load from cache (if just output settings were changed)\n 'fresh' to load from RDBMS \npyjade: ")

    # Load from cache
    if cache_or_fresh != 'fresh':
        print("Using cached data set")
        with Cache('DatasetCache') as ref:
            dataset = ref[f"{part_or_full}_cached"]

    # Get everything fresh from RDBMS
    else:
        print("Getting new data from RDBMS")
        dataset = Dataset()
        dataset.ingest(limit=limit)

        dataset.add_relations(limit=limit)

        dataset.quantify()
        dataset.unwrap_nonrepeating_columns()

        if cache_or_fresh == 'fresh':
            with Cache('DatasetCache') as ref:
                ref[f"{part_or_full}_cached"] = dataset

    # When using cache, you can optionally requantify things (helpful for development)
    if len(sys.argv) > 1:
        if sys.argv[1] == 'requantify':
            dataset.quantify()

    # Do you want quant info?
    if DATASET_OPTIONS['OUTPUT_STATS']:
        dataset.export_stats()

    # Do you want to run any data exports?
    if DATASET_OPTIONS['EXPORT']:

        if DATASET_OPTIONS['EXPORT_EVERYTHING_IN_SINGLE_XLSX_SHEET']:
            dataset.export_single_sheet()

        if DATASET_OPTIONS['EXPORT_COMPLETE_CURATED_DATASET']:
            dataset.export_complete_dataset()

        if DATASET_OPTIONS['EXPORT_SUBSETS']:
            dataset.export_subsets()

    # Do you want the noise / crumbs?
    if CRUMBS:
        dataset.export_crumbs()

    end = datetime.datetime.now()
    print("Time elapsed:",end-BEGIN)
