import streamlit as st
import pandas as pd
import numpy as np
import io
import joblib
from sentence_transformers import SentenceTransformer, util

st.set_page_config(layout="wide")

st.image('Bupa Logo.png', width = 100)

st.title('Kurum ve Kurum Tipi Eşlemesi')

st.write('\n')
st.write('\n')

st.markdown("""
    <div style="font-size: 20px; font-weight: bold;">
        Maplemenin yapılabilmesi için yüklenen dosya aşağıdaki gibi olmalıdır:
    </div>
    <div style="font-size: 18px;">
        <ul>
            <li>Dosya CSV UTF-8 formatında olmalıdır.</li>
            <li>Dosyanın ilk sütununda kurum isimleri olmalıdır ve A1'den başlamalıdır.</li>
            <li>Dosyanın diğer sütunlarında toplanması istenilen değerler olmalıdır.</li>
            <li>❌ Dosyada toplam tutar gösteren bir satır <b>olmamalıdır ❌</b>.</li>
            <li>Sayı gösteren sütunlarda <b>ondalık virgul (,) ile gösterilmelidir<b>. Örnek olarak: 1200,25 ✅ | 1200.25 ❌</li>
            <li>1000'den büyük sayılarda ayraç kullanılmamalıdır. Örnek olarak: 1200,25 ✅ | 1.200,25 ❌</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

st.write('\n')
st.markdown("""
    <div style="font-size: 20px; font-weight: bold;">
        Doğru bir CSV dosyası aşağıdaki gibi olmalıdır:
    </div>
""", unsafe_allow_html=True)
st.write('\n')
st.image('dogru csv.png')

st.write('\n')

@st.cache_resource
def load_references():
    return joblib.load('reference_data_new.pkl')

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def button_clicked_to_show():
    st.session_state['df_map_status'] += 1

def button_clicked_to_hide():
    st.session_state['df_map_status'] += 1

def clean_text(text):
    replacements = {
        '(eski Adı Biruni Üniv.sağ.eğitimi Uygulama Ve Araş. Merk)': '',
        "İ": "I", 
        "Ö": "O", 
        "Ü": "U", 
        "Ç": "C", 
        "Ş": "S", 
        "Ğ": "G",
        "?": " ", 
        "!": " ", 
        "-": " ", 
        "_": " ", 
        "  ": " ",
        " N.HAST": " FLORENCE NIGHTINGALE HASTANESI",
        "(ACIBADEM POLIKLINIKLERI A.S.)": "",
        "HASTANESI": "HOSPITAL", 
        "HASTANE": "HOSPITAL",
        "LIV HASTANESI": "LIV HOSPITAL", 
        "DR.": "DOKTOR ",
        "HİZ.TİC.A.Ş": "", 
        "HIZ.TIC": "", 
        " HIZ.": "", 
        " TIC.": "", 
        " AS.": "",
        " SAG ": " SAGLIK ", 
        " SAG.": " SAGLIK ",
        "ORTOPEDI": "ORT.", 
        "OZEL ": "", 
        " OZEL ": "",
        " ECZANE ": " ECZANESI ", 
        "ECZANE ": "ECZANESI ", 
        "ECZ.": "ECZANESI ",
        " HAS.": " HOSPITAL", 
        " HAS": " HOSPITAL ", 
        " HAST.": " HOSPITAL",
        " UNIVERSITE ": " UNIVERSITESI ", 
        " UNI ": " UNIVERSITESI ",
        " UNI. ": " UNIVERSITESI ", 
        " UNV ": " UNIVERSITESI ",
        " UNV. ": " UNIVERSITESI ", 
        " UNIV ": " UNIVERSITESI ",
        " UNIV. ": " UNIVERSITESI ", 
        " FTR ": " FIZIK TEDAVI VE REHABILITASYON ",
        "TC.": "", 
        "IST.": "ISTANBUL ", 
        "EGT VE ART.": "EGITIM VE ARASTIRMA ",
        "TIP FAK.": " ", 
        "MRK.": "MERKEZI ", 
        "  ": " "
    }
    
    for key, value in replacements.items():
        text = text.replace(key, value)
    
    return text.strip()

def matched_function(row):
    hc_group = hc_groups[row['kurum_adi']]
    return row['kurum_adi'], 100, hc_group


def unmatched_function(row, reference_dict, encoded_references):  
    category = "OTHER"
    for keyword in reference_dict.keys():
        if keyword in row["kurum_adi"]:
            category = keyword
            break

    filtered_references = reference_dict[category]
    filtered_embeddings = encoded_references[category]

    if not filtered_references or not len(filtered_embeddings):
        st.warning(f"No references found for category: {category}, kurum_adi: {row['kurum_adi']}")
        return None, 0, None

    input_embedding = model.encode(row["kurum_adi"], convert_to_tensor=True)

    cosine_scores = util.cos_sim(input_embedding, filtered_embeddings)

    best_idx = cosine_scores.argmax().item()
    best_score = cosine_scores[0, best_idx].item() * 100  
    best_match = filtered_references[best_idx]

    hc_group = hc_groups.get(best_match, None)

    return best_match, best_score, hc_group

def create_input_df(original_input):

    df = original_input.copy()

    numerical_columns = df.columns[1:].tolist()
    for col in numerical_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('.', '').str.replace(',', '.'), errors='coerce')

    if df.columns[0] == 'kurum_adi':
        df = df.rename(columns = {df.columns[0]: 'kurum_adi_'})
    
    df['kurum_adi'] = df[df.columns[0]].str.strip().str.upper().apply(clean_text) 
            
    return numerical_columns, df

def generate_excel(df, sheet_name):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            writer.close()
        return output.getvalue()

st.markdown("""
    <label style="font-size: 22px; font-weight: bold;">
        Kurum Adlarını Buraya Yükleyebilirsiniz:
    </label>
    """, unsafe_allow_html=True)

input_file = st.file_uploader('', type=['csv'])

if input_file is None:

    st.session_state['mapping_done'] = False
    st.session_state['df_map'] = None
    st.session_state['df_output'] = None
    st.session_state['df'] = None
    st.session_state['df_map_status'] = 1

if input_file is not None:

    original_input = pd.read_csv(input_file, delimiter=";") # yeni 07-03-2025
    original_input = original_input.dropna(axis=1, how="all") # yeni 07-03-2025
    original_input = original_input.dropna(subset=[original_input.columns[0]]) # yeni 07-03-2025

    with st.expander("Yüklemiş Olduğunuz Dosya", expanded=False):
        st.dataframe(original_input)

    if not st.session_state['mapping_done']:
        
        with st.spinner("Mapleme Yapılıyor"):    

            numerical_columns, df = create_input_df(original_input)
            model = load_model()
            hc_groups, reference_embeddings, reference_dict, encoded_references = load_references()
            reference_list = list(hc_groups.keys())

            df[["HEALTHCENTERDESC", "similarity_score", "KURUM GRUBU"]] = df.apply(
                    lambda row: pd.Series(
                        matched_function(row) if row["kurum_adi"] in reference_list else unmatched_function(row, reference_dict, encoded_references)
                    ),
                    axis=1
                )
            st.session_state['df'] = df

            df_output = (
                df.groupby('KURUM GRUBU', as_index=False)[numerical_columns].sum()
                .rename(columns = {col: f'TOTAL {col}' for col in numerical_columns})
                )
            
            st.success("Mapleme Tamamlandı")

            st.dataframe(df_output)
            st.session_state['df_output'] = df_output
            st.session_state['df_output_excel'] = generate_excel(st.session_state['df_output'], "Summary Data")

            st.session_state['mapping_done'] = True

            st.download_button(
                label="Özet Veriyi Excel Olarak İndir",
                data=st.session_state['df_output_excel'],
                file_name="ozet_veri.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            show_mapping_button = st.button('Kurumların Maplemesini Göster', key = 'show_button_1', on_click = button_clicked_to_show)

            df_map = (
                df[[
                    df.columns[0],
                    'kurum_adi',
                    'HEALTHCENTERDESC',
                    'KURUM GRUBU',
                    'similarity_score',
                ]]
                .rename(columns = {
                    'kurum_adi':f'{df.columns[0]} DUZENLENMIS',
                    'HEALTHCENTERDESC':'MAPLENDIGI KURUM ADI',
                    'KURUM GRUBU': 'MAPLENDIGI KURUM GRUBU ',
                    'similarity_score':'YAKINLIK SKORU',
                    }
                )
                .assign(MANUEL_KONTROL = lambda x: np.where(x['YAKINLIK SKORU'] <80, 'EVET', 'HAYIR'))
            )

            st.session_state['df_map'] = df_map
            st.session_state['df_map_excel'] = generate_excel(st.session_state['df_map'], "Mapped Data")

    elif st.session_state['df_map_status'] % 2 == 0:
        
        df = st.session_state['df']
        
        df_output = st.session_state['df_output']


        df_map = st.session_state['df_map']

        st.success("Mapleme Tamamlandı")
        st.dataframe(df_output)

        st.download_button(
            label="Özet Veriyi Excel Olarak İndir",
            data=st.session_state['df_output_excel'],
            file_name="ozet_veri.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.write('Maplenmiş Kurum Tablosu')
        st.dataframe(df_map)

        st.download_button(
            label="Haritalanmış Kurumları Excel Olarak İndir",
            data=st.session_state['df_map_excel'],
            file_name="maplenmis_kurumlar.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        show_mapping_button = st.button('Kurumların Maplemesini Gizle', key = 'hide_button', on_click = button_clicked_to_hide)

    else:

        st.success("Mapleme Tamamlandı")
        st.dataframe(st.session_state['df_output'])


        st.download_button(
            label="Özet Veriyi Excel Olarak İndir",
            data=st.session_state['df_output_excel'],
            file_name="ozet_veri.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        hide_mapping_button = st.button('Kurumların Maplemesini Göster', key = 'show_button_2', on_click = button_clicked_to_show)