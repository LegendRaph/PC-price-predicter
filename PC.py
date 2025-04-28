import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from rapidfuzz import process 


df = pd.read_csv('laptop_price.csv', encoding='latin-1')


df['Ram'] = df['Ram'].str.replace('GB', '').str.strip()
df['Ram'] = pd.to_numeric(df['Ram'], errors='coerce')

def extract_memory_types(mem_str):
    mem_str = str(mem_str).lower()
    ssd = 0
    hdd = 0
    if 'ssd' in mem_str:
        ssd_part = mem_str.split('ssd')[0]
        ssd = int(''.join(filter(str.isdigit, ssd_part)))
    if 'hdd' in mem_str:
        if '+' in mem_str:
            hdd_part = mem_str.split('+')[-1]
        else:
            hdd_part = mem_str
        hdd = int(''.join(filter(str.isdigit, hdd_part)))
    return pd.Series([ssd, hdd])

df[['SSD', 'HDD']] = df['Memory'].apply(extract_memory_types)


comp = ['Apple', 'HP', 'Acer', 'Asus', 'Dell', 'Lenovo', 'Chuwi', 'MSI',
        'Microsoft', 'Toshiba', 'Huawei', 'Xiaomi', 'Vero', 'Razer',
        'Mediacom', 'Samsung', 'Google', 'Fujitsu', 'LG']

res = ['IPS Panel Retina Display 2560x1600', '1440x900',
       'Full HD 1920x1080', 'IPS Panel Retina Display 2880x1800',
       '1366x768', 'IPS Panel Full HD 1920x1080',
       'IPS Panel Retina Display 2304x1440',
       'IPS Panel Full HD / Touchscreen 1920x1080',
       'Full HD / Touchscreen 1920x1080',
       'Touchscreen / Quad HD+ 3200x1800',
       'IPS Panel Touchscreen 1920x1200', 'Touchscreen 2256x1504',
       'Quad HD+ / Touchscreen 3200x1800', 'IPS Panel 1366x768',
       'IPS Panel 4K Ultra HD / Touchscreen 3840x2160',
       'IPS Panel Full HD 2160x1440', '4K Ultra HD / Touchscreen 3840x2160',
       'Touchscreen 2560x1440', '1600x900', 'IPS Panel 4K Ultra HD 3840x2160',
       '4K Ultra HD 3840x2160', 'Touchscreen 1366x768',
       'IPS Panel Full HD 1366x768', 'IPS Panel 2560x1440',
       'IPS Panel Full HD 2560x1440', 'IPS Panel Retina Display 2736x1824',
       'Touchscreen 2400x1600', '2560x1440', 'IPS Panel Quad HD+ 2560x1440',
       'IPS Panel Quad HD+ 3200x1800', 'IPS Panel Quad HD+ / Touchscreen 3200x1800',
       'IPS Panel Touchscreen 1366x768', '1920x1080',
       'IPS Panel Full HD 1920x1200', 'IPS Panel Touchscreen / 4K Ultra HD 3840x2160',
       'IPS Panel Touchscreen 2560x1440', 'Touchscreen / Full HD 1920x1080',
       'Quad HD+ 3200x1800', 'Touchscreen / 4K Ultra HD 3840x2160',
       'IPS Panel Touchscreen 2400x1600']


company_encoder = OrdinalEncoder(categories=[comp], dtype=int)
screenres_encoder = OrdinalEncoder(categories=[res], dtype=int)
product_encoder = LabelEncoder()


df['Company_enc'] = company_encoder.fit_transform(df[['Company']])
df['ScreenRes_enc'] = screenres_encoder.fit_transform(df[['ScreenResolution']])
df['Prod_enc'] = product_encoder.fit_transform(df['Product'])


X = df[['Ram', 'SSD', 'HDD', 'Company_enc', 'Prod_enc', 'ScreenRes_enc']]
y = df['Price_in_euros']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)


def fuzzy_correct(user_input, valid_options, field_name='field', cutoff=70):
    match = process.extractOne(user_input, valid_options, score_cutoff=cutoff)
    if match:
        return match[0]
    else:
        raise ValueError(f"No close match found for {field_name}: '{user_input}'.")


print("\nExample input:\n - 8GB RAM, 512GB SSD, Dell, XPS 13, Full HD 1920x1080")
user_input = input("Enter Laptop Specifications: ")


try:
    parts = [x.strip() for x in user_input.split(',')]

 
    ram_str = parts[0]
    ram_input = int(ram_str.lower().replace('gb ram', '').strip())


    ssd_input = 0
    hdd_input = 0

    if 'ssd' in parts[1].lower():
        ssd_str = parts[1]
        ssd_input = int(ssd_str.lower().replace('gb ssd', '').replace('tb ssd', '000').strip())
      
        company_input, product_input, screenres_input = parts[2], parts[3], parts[4]
    elif 'hdd' in parts[1].lower():
        hdd_str = parts[1]
        hdd_input = int(hdd_str.lower().replace('gb hdd', '').replace('tb hdd', '000').strip())
       
        company_input, product_input, screenres_input = parts[2], parts[3], parts[4]
    else:
        raise ValueError("Second input must specify SSD or HDD.")

  
    company_corrected = fuzzy_correct(company_input, comp, field_name='Company')
    screenres_corrected = fuzzy_correct(screenres_input, res, field_name='ScreenResolution')


    company_encoded = company_encoder.transform([[company_corrected]])[0][0]
    screenres_encoded = screenres_encoder.transform([[screenres_corrected]])[0][0]

  
    product_list = list(product_encoder.classes_)
    product_corrected = fuzzy_correct(product_input, product_list, field_name='Product', cutoff=60)


    product_encoded = product_encoder.transform([product_corrected])[0]

    
    user_features = [[ram_input, ssd_input, hdd_input, company_encoded, product_encoded, screenres_encoded]]

    predicted_price = model.predict(user_features)
    print(f"\n💻 Estimated Laptop Price: €{predicted_price[0]:.2f}")

except Exception:
    print("\n⚠️ Error! Please check your input carefully and enter valid system specifications.")
