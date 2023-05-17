#########################
"""
This code and its contents are the property of the author, Eric Brattin, alone 
and do not represent any group or organization, whether affiliated or not.
 
Any views or opinions expressed in this code are solely those of the author and do not reflect 
the views or opinions of any group or organization. 

This code is provided as-is, without warranty of any kind, express or implied.
"""
#########################
import importlib
import subprocess
import sys
import pkg_resources
import pandas as pd

packages = [
    ("aiofiles",),
    ("aiohttp",),
    ("altair", "alt"),
    ("asyncio",),
    ("chardet",),
    ("codecs",),
    ("colorama",),
    ("concurrent",),
    ("concurrent.futures", None, ["ThreadPoolExecutor", "as_completed", "ProcessPoolExecutor"]),
    ("csv",),
    ("dask", None, ["delayed"]),
    ("dask.dataframe", "dd"),
    ("datetime", "dt", ["date"]),
    ("dill",),
    ("fastparquet",),
    ("functools", None, ["partial"]),
    ("fuzzywuzzy", None, ["fuzz"]),
    ("glob",),
    ("hashlib",),
    ("importlib",),
    ("io",),
    ("missingno", "msno"),
    ("nest_asyncio",),
    ("numpy", "np"),
    ("os",),
    ("PIL", None, ["Image"]),
    ("pandas", "pd"),
    ("pathlib",),
    ("plotly.express", "px"),
    ("plotly.graph_objects", "go"),
    ("rapidfuzz", None, ["process"]),
    ("re",),
    ("requests",),
    ("spacy",),
    ("sqlalchemy", None, ["create_engine", "text"]),
    ("sqlalchemy==1.4.27",),
    ("subprocess",),
    ("sys",),
    ("tabulate", "tabulate"),
    ("tabulate",),
    ("tempfile",),
    ("termcolor", None, ["colored"]),
    ("textacy",),
    ("time",),
    ("tqdm",),
    ("tqdm.auto", None, ["tqdm"]),
    ("translate", None, ["Translator"]),
    ("urllib",),
    ("webbrowser",),
]

def import_and_install(package):
    try:
        return importlib.import_module(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return importlib.import_module(package)

def update_globals(pkg, alias=None, submodules=None):
    try:
        if alias:
            module = import_and_install(pkg)
            if isinstance(alias, tuple):
                for a in alias:
                    globals()[a] = module
            else:
                globals()[alias] = module
        else:
            module = import_and_install(pkg)

        if submodules:
            for submodule in submodules:
                globals()[submodule] = getattr(module, submodule)
        return module

    except AttributeError:
        print(f"Error: The '{pkg}' package does not have the specified submodule(s).")
        if pkg == "fuzzywuzzy":
            print("Trying to upgrade or reinstall the 'fuzzywuzzy' package...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "fuzzywuzzy"])
            except subprocess.CalledProcessError:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "--no-cache-dir", "fuzzywuzzy"])
    except Exception as e:
        print(f"Error: {e}. Failed to install the '{pkg}' package.")

installed_packages = {pkg.key for pkg in pkg_resources.working_set
installed_packages = sorted(list(installed_packages))


modules = []
for package in packages:
    if isinstance(package, tuple):
        try:
            pkg, alias, submodules = package
        except ValueError:
            pkg, alias = package
            submodules = None
    else:
        pkg, alias, submodules = package, None, None

    if pkg in installed_packages:
        try:
            module = importlib.import_module(pkg)
            if alias:
                if isinstance(alias, tuple):
                    alias_name = alias[0]
                else:
                    alias_name = alias

                modules.append({"module": pkg, "alias": alias_name})
            else:
                modules.append({"module": pkg, "alias": None})

            if submodules:
                for submodule in submodules:
                    globals()[submodule] = getattr(module, submodule)
        except Exception as e:
            print(f"Error: {e}. Failed to import the '{pkg}' package.")

module_df = pd.DataFrame(modules)

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

if not spacy.util.is_package("en_core_web_md"):
    try:
        install("spacy[en_core_web_md]")
    except subprocess.CalledProcessError:
        spacy_version = spacy.__version__
        install(f"spacy=={spacy_version}")
        install("spacy[en_core_web_md]")
    except Exception as e:
        print(f"Error installing 'en_core_web_md': {e}")
        sys.exit(1)

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "en_core_web_md"])
    except subprocess.CalledProcessError:
        pass

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "en_core_web_md"])
    except subprocess.CalledProcessError:
        pass
    conflicting_packages = ["spacy", "thinc"]
    for package in conflicting_packages:
        if package in sys.modules:
            del sys.modules[package]
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "en_core_web_md"])
    except subprocess.CalledProcessError:
        pass
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.1.0/en_core_web_md-3.1.0-py3-none-any.whl"])
    except subprocess.CalledProcessError:
        pass

nlp = spacy.load('en_core_web_md')


table = []
for package in packages:
    if isinstance(package, str):
        module = __import__(package)
    elif isinstance(package, tuple):
        module = __import__(package[0])
    else:
        continue
    module_name = module.__name__.split(".")[-1]
    module_attributes = [attribute for attribute in dir(module) if not attribute.startswith("_")]
    for attribute in module_attributes:
        table.append([module_name, attribute])
print(tabulate(table, headers=["Module", "Sub-Module"], tablefmt="psql"))

path_desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
path_aid = os.path.join(path_desktop, 'USAID')
path_data=os.path.join(path_aid, 'Data')
parquet_path = os.path.join(path_aid, 'Parquet')

if not os.path.exists(path_aid):
    os.makedirs(path_aid)
    print("Created folder: USAID")
else:
    print("USAID folder already exists")
if not os.path.exists(path_data):
    os.makedirs(path_data)
    print("Created folder: Data")
else:
    print("Data folder already exists")
    
os.chdir(path_data)
def excel_to_df(url):
    response = requests.get(url)
    with open("temp.xlsx", "wb") as f:
        f.write(response.content)
    with pd.ExcelFile("temp.xlsx") as xlsx:
        dataframes = {sheet.replace(' ', '_'): xlsx.parse(sheet) for sheet in xlsx.sheet_names}
    for sheet_name, df in dataframes.items():
        globals()[sheet_name] = df
        print(f"\nSheet name: {sheet_name}")
        print(tabulate(df.iloc[:10, :3], headers='keys', tablefmt='psql'))

rol_data = "https://worldjusticeproject.org/rule-of-law-index/downloads/FINAL_2022_wjp_rule_of_law_index_HISTORICAL_DATA_FILE.xlsx"
excel_to_df(rol_data)


global_vars = globals().copy()
for name, obj in global_vars.items():
    if isinstance(obj, pd.DataFrame) and re.search(r'_\d{4}_', name):
        year = name.split('_')[3][-2:]
        exec(f"rol_{year} = {name}")
        del globals()[name]

df_dict = {'rol_19': rol_19, 'rol_20': rol_20, 'rol_21': rol_21, 'rol_22': rol_22}

def get_column_name(text, pos):
    doc = nlp(text.lower())
    significant_words = []
    for token in doc:
        if not token.is_punct and token.pos_ in ['NOUN', 'VERB', 'ADJ']:
            if token.text == 'effectively':
                continue
            significant_words.append(token.text)
    limited_idx = -1
    by_idx = -1
    for i, word in enumerate(significant_words):
        if word == 'limited':
            limited_idx = i
        elif word == 'by':
            by_idx = i
    if limited_idx >= 0 and by_idx > limited_idx:
        significant_words.remove('by')
        significant_words[limited_idx] = 'limited by judiciary'
    new_col_name = ' '.join(significant_words)
    if new_col_name.strip() == '':
        new_col_name = 'column' + str(pos + 1)
    return new_col_name

for name, df in df_dict.items():
    try:

        df = df.transpose()
        df.columns = df.iloc[0]  
        df = df.iloc[1:]
        df.dropna(axis=1, how='all', inplace=True)
        df.reset_index(inplace=True)


        pattern = r'(0\.[0-9]+|[1-9]\d*\.\d+)'
        for col in df.columns:
            if any(re.search(pattern, str(x)) for x in df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce').round(3)

        def map_column_name(col):
            if 'Colombia' in df[col].values:
                return 'Country'
            elif df[col].astype(str).str.contains(r'\bATG\b|\bDZA\b|\bBGD\b', case=True).any():

                return 'Country Code'
            else:
                return col
        rename_dict = {col: map_column_name(col) for col in df.columns}
        df = df.rename(columns=rename_dict)
        df.columns = [re.sub(r'^.*:\s*|\d+\.\d+\s', '', col) for col in df.columns]
        df.columns = df.columns.str.title()

        num_rows_dropped = len(df) - df.dropna().shape[0]
        print(f"Processed {name}: Dropped {num_rows_dropped} rows, and cleaned column names.")

        globals()[name] = df

    except KeyError:
        print(f"Skipping {name} because it is not in the dictionary.")
    except Exception as e:
        print(f"An error occurred while processing {name}: {e}")


url_country_code ="https://wits.worldbank.org/wits/wits/witshelp/content/codes/country_codes.htm"

with urllib.request.urlopen(url_country_code) as i:
    html = i.read()
code_df = pd.read_html(html)[0]
print(code_df.head())

code_df.columns = code_df.iloc[0]
code_df = code_df.iloc[2:]

pattern = re.compile(r'^\d+$')
if code_df.iloc[:, -1].dtype == 'object' and code_df.iloc[:, -1].str.match(pattern).all():
    code_df = code_df.iloc[:, :-1]

url_dem="https://www.democracymatrix.com/ranking"

with urllib.request.urlopen(url_dem) as i:
    html = i.read()
dem_df = pd.read_html(html)[0]
print(dem_df.head())


def get_best_match(country_name, choices):
    return process.extractOne(country_name, choices)[0]

dem_df['Country'] = dem_df['Country'].apply(get_best_match, args=[code_df['Country Name']])
dem_code_df = dem_df.merge(code_df, left_on="Country", right_on="Country Name", how = 'inner')

dem_code_df['Country'] = dem_code_df['Country'].apply(get_best_match, args=[rol_22['Country']])

rol_years = {19: rol_19, 20: rol_20, 21: rol_21, 22: rol_22}

modified_rol_years = {'rol_' + str(key): value for key, value in rol_years.items()}

rol_dem_cmb = {}

for name, rol_df in modified_rol_years.items():
    year = int(name.split('_')[1])
    rol_dem_cmb[name] = dem_code_df.merge(rol_df, left_on="Country Codes", right_on="Country Code", how='inner')
    rol_dem_cmb[name].rename(columns=lambda x: re.sub(r'^\d+\.\d+\.\s+', '', x), inplace=True)
    print(rol_dem_cmb[name].columns)
    
for name, df in rol_dem_cmb.items():
    globals()[name] = df


def clean_column_names(df):
    new_cols = []
    sim_cols = []
    for col in df.columns:
        col_cleaned = col.strip().lower().replace(' ', '_')
        for new_col in new_cols:
            if nlp(col_cleaned).similarity(nlp(new_col)) >= 0.9:
                sim_cols.append((new_col, col_cleaned))
                col_cleaned = f"{new_col}_{col_cleaned}"
                break
        new_cols.append(col_cleaned)

    cleaned_cols = dict(zip(df.columns, new_cols))
    if len(set(cleaned_cols.values())) < len(cleaned_cols):
        print("Duplicate column names found after renaming!")
        print(cleaned_cols)
    
    return cleaned_cols

ecuador_dfs = []
for year in ['19', '20', '21', '22']:
    df = globals().get('rol_' + year)
    if isinstance(df, pd.DataFrame):
        ecuador_df = df[df['Country Name'] == 'Ecuador']
        ecuador_df['Year'] = year
        ecuador_df = ecuador_df.rename(columns=clean_column_names)
        ecuador_dfs.append(ecuador_df)
    else:
        print(f"rol_{year} is not a DataFrame and will be skipped.")

if ecuador_dfs:
    ecuador_all = pd.concat(ecuador_dfs, ignore_index=True)
else:
    ecuador_all = pd.DataFrame()
    print("No valid DataFrames found.")



float_columns = [col for col in ecuador_all.columns if ecuador_all[col].dtype == np.float64]
percentage_change = {'Year': 'Percentage Change'}
for col in float_columns:
    year_19_val = ecuador_all.loc[ecuador_all['Year'] == '19', col].values[0]
    year_22_val = ecuador_all.loc[ecuador_all['Year'] == '22', col].values[0]
    
    if year_19_val != 0:
        change = ((year_22_val - year_19_val) / year_19_val) * 100
    else:
        change = np.nan
    percentage_change[col] = change
ecuador_all = ecuador_all.append(percentage_change, ignore_index=True)


negative_changes = {k: v for k, v in percentage_change.items() if isinstance(v, float) and v < 0}
negative_changes_ec = pd.DataFrame(negative_changes, index=[0]).transpose()
negative_changes_ec.reset_index(inplace=True)
negative_changes_ec.columns = ['Description', 'Percentage Change']

from googletrans import Translator
def translate_to_spanish(text):
    try:
        translator = Translator(to_lang="es")
        translated = translator.translate(text)
        return translated
    except:
        return text

negative_changes_ec['Description_ES'] = negative_changes_ec['Description'].apply(translate_to_spanish)

bar_chart = alt.Chart(negative_changes_ec).mark_bar().encode(
    x=alt.X('Description:N', sort='-y', title='Description'),
    y=alt.Y('Percentage Change:Q', title='Percentage Change'),
    color=alt.condition(
        alt.datum['Percentage Change'] > 0,
        alt.value('steelblue'),
        alt.value('darkblue')
    )
)
text = bar_chart.mark_text(
    align='center',
    baseline='bottom',
    dy=-5
).encode(
    text=alt.Text('Percentage Change:Q', format='.1f')
)
combined_chart = bar_chart + text
html = combined_chart.save("combined_chart.html")
webbrowser.open('file://' + os.path.realpath("combined_chart.html"))


sorted_data = negative_changes_ec.sort_values(by='Percentage Change', ascending=True)
colors = px.colors.sequential.Blues
color_step = len(colors) // len(sorted_data)
color_scale = [colors[i*color_step] for i in range(len(sorted_data))]

fig = px.bar(sorted_data, x='Description', y='Percentage Change', text='Percentage Change',
             color='Percentage Change', color_discrete_sequence=color_scale,
             title='Negative Percentage Changes in Ecuador')

fig.update_layout(showlegend=False)
fig.update_traces(texttemplate='%{text:.2f}', textposition='inside')

fig.write_html('Negative Percentage Changes in Ecuador.html')

webbrowser.open('file://' + os.path.realpath("Negative Percentage Changes in Ecuador.html"))


positive_changes = {k: v for k, v in percentage_change.items() if isinstance(v, float) and v > 0}

positive_changes_ec = pd.DataFrame(positive_changes, index=[0]).transpose()
positive_changes_ec.reset_index(inplace=True)
positive_changes_ec.columns = ['Description', 'Percentage Change']

positive_changes_ec['Description_ES'] = positive_changes_ec['Description'].apply(translate_to_spanish)

os.chdir(path_aid)

start_time = time.time()

nest_asyncio.apply()

urls_and_filenames = [
    ("https://files.usaspending.gov/award_data_archive/FY2022_All_Contracts_Full_20230407.zip", "Treas_22.zip"),
    ("https://files.usaspending.gov/award_data_archive/FY2021_All_Contracts_Full_20230407.zip", "Treas_21.zip"),
    ("https://files.usaspending.gov/award_data_archive/FY2020_All_Contracts_Full_20230407.zip", "Treas_20.zip"),
    ("https://files.usaspending.gov/award_data_archive/FY2019_All_Contracts_Full_20230407.zip", "Treas_19.zip")]

usaspend_cols = ['transaction_number', 'award_id_piid', 'action_date_fiscal_year',
                 'treasury_accounts_funding_this_award', 'program_activities_funding_this_award',
                 'award_or_idv_flag', 'award_type', 'product_or_service_code_description',
                 'object_classes_funding_this_award', 'parent_award_agency_name',
                 'parent_award_id_piid', 'current_total_value_of_award',
                 'period_of_performance_start_date', 'period_of_performance_current_end_date',
                 'awarding_agency_name', 'awarding_sub_agency_name', 'awarding_office_name',
                 'funding_agency_name', 'funding_sub_agency_name', 'recipient_duns',
                 'recipient_uei', 'recipient_name', 'recipient_parent_duns', 'recipient_parent_uei',
                 'recipient_parent_name', 'recipient_country_name',
                 'primary_place_of_performance_country_name',
                 'primary_place_of_performance_city_name',
                 'type_of_contract_pricing', 'parent_award_single_or_multiple']

date_columns = ['period_of_performance_start_date',
                'period_of_performance_current_end_date']



def download_file(url, filename):
    file = os.path.join(path_data, filename)
    hash_file = os.path.join(path_data, f"{filename}.md5")
    print(f"Checking if {filename} exists...")
    if os.path.exists(file):
        if os.path.exists(hash_file):
            with open(hash_file, 'r') as f:
                file_md5 = f.read().strip()
        else:
            with open(file, 'rb') as f:
                file_md5 = hashlib.md5(f.read()).hexdigest()
                with open(hash_file, 'w') as h:
                    h.write(file_md5)
        with requests.get(url, stream=True, timeout=(1200, 1200)) as response:
            response_md5 = hashlib.md5(response.content).hexdigest()
        if file_md5 == response_md5:
            print(f"{filename} exists and is the same as the downloaded file.")
            return
        else:
            os.remove(file)
            os.remove(hash_file)
            print(f"{filename} exists but is not the same as the downloaded file. Deleting the existing file and downloading the new file.")
    else:
        print(f"{filename} does not exist. Downloading the file...")

    with requests.get(url, stream=True, timeout=(1200, 1200)) as response:
        total_size = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
        with open(file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
        with open(hash_file, 'w') as h:
            h.write(hashlib.md5(open(file, 'rb').read()).hexdigest())
    print(f"{filename} downloaded successfully.")
    progress_bar.close()


def extract_and_rename(filename):
    if not filename.endswith('.zip'):
        print(f"{filename} is not a ZIP file.")
        return

    try:
        with zipfile.ZipFile(os.path.join(data_path, filename), mode="r") as archive:
            file_contents = [(i.filename, i.file_size) for i in archive.infolist()]
            print(f"Contents of {filename}:")
            print(tabulate(file_contents))
            archive.extractall(data_path)
            extracted_files = os.listdir(data_path)
            csv_files = [f for f in extracted_files if f.endswith('.csv')]
            for i, csv_file in enumerate(csv_files, start=1):
                new_filename = f"{filename.split('.')[0][-2:]}_{i}.csv"
                os.rename(os.path.join(data_path, csv_file), os.path.join(ipeds_path, new_filename))

            print(f"Successfully extracted and renamed {len(csv_files)} CSV files from {filename}.")
    except zipfile.BadZipFile as error:
        print(f"Error extracting {filename}: {error}")


def test_extract_and_rename():
    zip_files = ['Treas_19.zip', 'Treas_20.zip', 'Treas_21.zip', 'Treas_22.zip']
    for zip_file in zip_files:
        extract_and_rename(zip_file)
        with zipfile.ZipFile(os.path.join(data_path, zip_file), mode="r") as archive:
            file_contents = [(i.filename, i.file_size) for i in archive.infolist()]
            print(f"Contents of {zip_file}:")
            print(tabulate(file_contents))


async def process_chunk(chunk, chunk_number, total_chunks):
    print(f"Processing chunk {chunk_number} of {total_chunks}...")
    chunk.rename(columns={col.lower(): col for col in chunk.columns}, inplace=True)

    chunk.columns = [x.title() for x in chunk.columns]

    unwanted_columns_pattern = re.compile('^(' + '|'.join(['(?i)unnamed', '(?i)empty']) + ')')
    chunk = chunk.loc[
        :, ~chunk.columns.str.match(unwanted_columns_pattern)
    ].replace(
        {'(?i)^no$': False, '(?i)^yes$': True}, regex=True
    ).replace(np.nan, "", regex=True)
    empty_columns = []

    for col in chunk.columns:
        if chunk[col].astype(str).str.strip().replace('', np.nan).isna().all():
            empty_columns.append(col)
    if empty_columns:
        chunk.drop(empty_columns, axis=1, inplace=True)

    print(f"Finished processing chunk {chunk_number} of {total_chunks}.")
    return chunk


def convert_numeric_columns(chunk):
    for col in chunk.columns:
        if col in date_columns:
            chunk[col] = pd.to_datetime(chunk[col], errors='coerce')
        elif chunk[col].str.isnumeric().all():
            chunk[col] = chunk[col].astype(int)
        elif chunk[col].str.replace('.', '', regex=False).str.isnumeric().all():
            chunk[col] = chunk[col].astype(float)
    return chunk


async def process_file(file):
    print(f"Processing file: {file}")
    df = pd.read_csv(os.path.join(path_data, file), dtype=str, chunksize=5000, encoding='latin-1')
    total_chunks = sum(1 for _ in pd.read_csv(os.path.join(path_data, file), dtype=str, chunksize=5000, encoding='latin-1'))

    unwanted_columns_pattern = f'(?i)(?:{"|".join(["unnamed", "empty"])})'

    processed_chunks = []

    for i, chunk in enumerate(df):
        chunk.columns = [x.replace('_', ' ').title() for x in chunk.columns]
        chunk = chunk.loc[:, ~chunk.columns.str.contains(unwanted_columns_pattern, case=False)]

        if len(chunk.columns) == 0:
            return None

        chunk = convert_numeric_columns(chunk)

        processed_chunk = process_chunk(chunk, i + 1, total_chunks)
        processed_chunks.append(processed_chunk)

    concatenated_df = pd.concat(processed_chunks)
    print(f"Finished processing file: {file}")
    return concatenated_df

async def download_and_extract_file(url, filename):
    print(f"Downloading {filename}...")
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, partial(download_file, url, filename))
    print(f"Finished downloading {filename}.")
    
    print(f"Extracting and renaming {filename}...")
    await loop.run_in_executor(None, partial(extract_and_rename, filename))
    print(f"Finished extracting and renaming {filename}.")


async def main():
    async with aiohttp.ClientSession() as session:
        tasks = []

        for url, filename in urls_and_filenames:
            tasks.append(asyncio.create_task(download_and_extract_file(url, filename)))

        await asyncio.gather(*tasks)
        print("Finished downloading and extracting all files.")

        processed_dfs = []

        for file in os.listdir(path_data):
            if file.endswith(".csv"):
        
                concatenated_df = await loop.run_in_executor(None, partial(process_file, file))
                processed_df = concatenated_df[usaspend_cols]
                processed_ddf = dd.from_pandas(processed_df, npartitions=10)
                processed_dfs.append(processed_ddf)

        if not processed_dfs:
            print("No DataFrames to concatenate.")
            return None
        else:

            concatenated_ddf = dd.concat(processed_dfs, axis=0)
            concatenated_df = concatenated_ddf.compute()

    return concatenated_df


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        result = loop.run_until_complete(main())
    except Exception as e:
        print(f"An error occurred during processing: {e}")
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()

    if result is not None:
        result.to_parquet(output_file, index=False, engine='fastparquet')
        print(f"Data has been written to {output_file}.")
    else:
        print("No data was processed. No output file was written.")
    test_extract_and_rename()

os.chdir(path_aid)   
filename = 'AID_DATA_DEMO'
ext = '.bz2'
filenametime = dt.datetime.now().strftime("%Y%m%d")                  
dill.dump_session(filename+filenametime+ext)

concatenated_df = concatenated_df.loc[:, ~uconcatenated_df.columns.duplicated()].copy()
concatenated_df = concatenated_df.reindex(sorted(concatenated_df.columns), axis=1)

raw_described = concatenated_df.describe(include='all').loc['unique', :]
raw_described = raw_described.reset_index().sort_values(by=['unique'], ascending=False)
raw_described['unique'] = raw_described['unique'].apply(lambda x: "{:,}".format(x))
print(tabulate(raw_described, tablefmt="grid"))


start = time.time()
engine = create_engine('sqlite:///USA_Spend.db', echo=True)
sqlite_connection = engine.connect()
sqlite_connection.execute(text("PRAGMA journal_mode = OFF"))
sqlite_table = "USA_Spend"
concatenated_df.to_sql(sqlite_table, sqlite_connection, if_exists='replace', index=False, chunksize=1000)
sqlite_connection.close()
end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("SQLite import took {:0>2}:{:0>2}:{:05.2f}".format(
    int(hours), int(minutes), seconds))

