import streamlit as st
import pandas as pd
import numpy as np
import plotly_express as px
import plotly.graph_objects as go
from PIL import Image
st.set_page_config(layout='wide')
import streamlit.components.v1 as components

# import plotly.io as pio
# pio.templates
# pio.templates.default = "simple_white"
# load model 
import joblib
import sklearn
import base64
from io import BytesIO
# linear programming
import pulp
from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable

def main():
    # st.title("Simulasi Realokasi Budget untuk memaksimalkan Efisiensi dan Omset")
#     greach= Image.open('GReach.png')
#     st.sidebar.image(greach)
    # menu = ["Sasaran Penerima Bantuan","Sebaran Program Pemerintah","Daftar Kandidat Penerima Bantuan","Pendaftaran UMKM"]
    menu = ['MSME Report and Analysis',"Government Benefits",'MSME Candidate List','OCA Dashboard']
    choice = st.sidebar.selectbox("Select Menu", menu)
    df = pd.read_excel('UMKM_Efisiensi.xlsx')
    # from gsheetsdb import connect

    # # Create a connection object.
    # conn = connect()

    # # Perform SQL query on the Google Sheet.
    # # Uses st.cache to only rerun when the query changes or after 10 min.
    # # @st.cache(ttl=600)
    # def run_query(query):
    #     rows = conn.execute(query, headers=1)
    #     return rows

    # sheet_url = st.secrets["public_gsheets_url"]
    # rows = run_query(f'SELECT * FROM "{sheet_url}"')

    # if choice == "Pendaftaran UMKM":
    #     for row in rows:
    #         st.write(f"{row.Timestamp}")
    #         # st.write(rows)
    #     st.text_input(label='Nama: ')
    #     # components.html(
    #     #     '''<iframe src="https://docs.google.com/forms/d/e/1FAIpQLSePcRyj3AM3GV0Yqpz0_29Xxhv2nhLk_MsgVJzritKPiWvVXw/viewform?embedded=true" width="640" height="809" frameborder="0" marginheight="0" marginwidth="0">Loading…</iframe>
    #     #     ''',
    #     # height=809,
    #     # width=640
    #     # )
    #     components.html(
    #         '''
    #     <div class="container" id="ff-compose"></div>
    #     <script async defer src="https://formfacade.com/include/109706827221839419224/form/1FAIpQLSePcRyj3AM3GV0Yqpz0_29Xxhv2nhLk_MsgVJzritKPiWvVXw/bootstrap.js?div=ff-compose"></script>
    #     ''',
    #     height=809,
    #     width=640
    #     )
    # elif choice == "Daftar Kandidat Penerima Bantuan":
    if choice == 'OCA Dashboard':
        components.html(
            '''
            <style>
              #wrap { width: 1020px; height: 900px; padding: 0; border: 0px solid grey; overflow: hidden; }
              #frame { width: 900px; height: 1980px; margin-left: -5px;margin-right: 0px; margin-top: -10px; }
              #frame {
                  -ms-zoom: 1;
                  -moz-transform: scale(1);
                  -moz-transform-origin: 0 0;
                  -o-transform: scale(1);
                  -o-transform-origin: 0 0;
                  -webkit-transform: scale(1);
                  -webkit-transform-origin: 0 0;
              }
              #frame2 { width: 1020px; height: 900px; border: 0px solid black; margin-top: -195px}
          </style>
          <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
              
          <div class="wrap-element" style="padding: 0; overflow: hidden;">
            <iframe id="frame" scrolling="no" class="wrapped-iframe" gesture="media"  allow="encrypted-media" allowfullscreen = "True"
            name="Framename" sandbox="allow-same-origin allow-scripts allow-popups allow-forms"
            src="https://dashboard.ocaindonesia.co.id/"
            style="width: 100%;">
            </iframe>
            ''',
        height=1150,
        width=1280
        )
    elif choice == 'MSME Candidate List':
        layak = []
        for k in df['Efisiensi'].tolist():
            if k>=0.85:
                layak.append('Eligible')
            else:
                layak.append('Ineligible')
        df['kelayakan'] = layak
        provinsi = st.selectbox('Province',df.Prov.unique())
        df = df[df['Prov'].isin([provinsi])]
        pemda = st.selectbox('City',df.Kab_APBD.unique())
        df = df[df['Kab_APBD'].isin([pemda])]
        dfp = pd.read_excel('UMKM_APBD.xlsx')
        dfp = dfp[dfp['Urpemda'].isin([pemda])]
        k1,k2 =st.beta_columns((3,2))
        with k2:
            st.write('Candidates Eligibility: ')
            st.write('')
            fig0 = px.pie(df,names='kelayakan',values='Efisiensi',hole=0.6)
            fig0.update_layout(width=400,height=500,margin=dict(l=15, r=0, t=20, b=20))
            st.plotly_chart(fig0)
        with k1:
            st.write('Available Programs: ')
            st.table(dfp['Kegiatan'])
            
        dflist = df[['BU','omset','Efisiensi','Nama_pasar','kelayakan']]
        dflist.columns = ['MSME_Name','Total_Revenue','Eficiency','Location','Eligibility']
        st.write('Candidate List: ')
        st.dataframe(dflist)
        
        dfk = df[df['Efisiensi']>=0.85]
        dfn = df[df['Efisiensi']<0.85]
        def to_excel(df):
            output = BytesIO()
            writer = pd.ExcelWriter(output, engine='xlsxwriter')
            df.to_excel(writer, sheet_name='Sheet1')
            writer.save()
            processed_data = output.getvalue()
            return processed_data

        def get_table_download_link(df,file,name):
            """Generates a link allowing the data in a given panda dataframe to be downloaded
            in:  dataframe
            out: href string
            """
            val = to_excel(df)
            b64 = base64.b64encode(val)  # val looks like b'...'
            return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{file}.xlsx">{name}</a>' # decode b'abc' => abc

        st.markdown(get_table_download_link(dfk,'Candidate List','Download Candidate List'), unsafe_allow_html=True)
        st.markdown(get_table_download_link(dfn,'Non-Candidate List','Download Non-Candidate List'), unsafe_allow_html=True)
    
    # elif choice == "Sebaran Program Pemerintah":
    elif choice == "Government Benefits":
        components.html(
            '''<div class='tableauPlaceholder' id='viz1629105909629' style='position: relative'><noscript><a href='#'><img alt='Dashboard Pemberdayaan UMKM ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Da&#47;DashboardUMKM&#47;Dashboard1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='DashboardUMKM&#47;Dashboard1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Da&#47;DashboardUMKM&#47;Dashboard1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1629105909629');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else { vizElement.style.width='100%';vizElement.style.height='1127px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>''',
        height=1150,
        width=1280
        )
    # elif choice == "Sasaran Penerima Bantuan":
    elif choice == 'MSME Report and Analysis':
        st.subheader('Eligibily Check And Efficiency Strategy Recommendation')
        provinsi = st.sidebar.selectbox('Province',df.Prov.unique())
        df = df[df['Prov'].isin([provinsi])]
        pemda = st.sidebar.selectbox('City',df.Kab_APBD.unique())
        df = df[df['Kab_APBD'].isin([pemda])]
        umkm = st.sidebar.selectbox('Choose MSME Name',['All']+df.BU.unique().tolist())
        st.title(umkm)
        if umkm=='All':
            st.write('Please Choose MSME')
        else:
            dff = df[df['BU'].isin([umkm])]
            st.subheader(f'Location: {dff.Nama_pasar.values[0]}')
            st.subheader(f'Eficiency Rate: {int(dff.Efisiensi.mean()*10000)/100} %')
            st.subheader(f'Total Revenue: Rp {int(dff.omset.sum()):,d}')
            # st.number_input(label='Omset UMKM',value=int(dff.omset.sum()), min_value=0,max_value=1000000000000)
            s1='Beban_UmumAdm'
            s2='Beban_Penjualan'
            s3='Beban_Lainnya'

            if int(dff.Efisiensi.mean()*100) >84.9:
                st.subheader('Eligible for Government Benefits')
                model= open("lbst_umk.pkl", "rb")
                huber=joblib.load(model)
                dfvalues = dff[['omset',s1,s2,s3,'Efisiensi']]
                # dfvalues = pd.DataFrame(list(zip([sl],[sw],[pl],[pw])),columns =['lengkap', 'tepatwaktu', 'ketsesuai', 'adapembayaran'])
                # input_variables = np.array(dfvalues[['lengkap', 'tepatwaktu', 'ketsesuai', 'adapembayaran']])
                input_variables = np.array(dfvalues)
                with st.beta_expander('Maximum Fund can be claimed', expanded=False):
                    if st.button('Prediction of monthly installment ability'):
                        prediction = huber.predict(input_variables)
                        st.title(f'Maximum Installment: Rp {int(prediction):,d}')
                        
            else:
                
                dfs = dff[[s1,s2,s3,'Total_beban','BU']]
                dfs[s1] = dfs[s1]/dfs['Total_beban']
                dfs[s2] = dfs[s2]/dfs['Total_beban']
                dfs[s3] = dfs[s3]/dfs['Total_beban']
                dfm = pd.melt(dfs,id_vars=['BU'],value_vars=[s1,s2,s3])
                dflp = df[df['Kab_APBD'].isin([pemda])]
                dflp = dflp.replace(to_replace=0,value=np.NAN)
                top = dflp['Efisiensi'].max()
                # st.write(top)
                dflp = dflp[dflp['Efisiensi']>=top-0.05]
                with st.beta_expander('Frontier MSME in the Area', expanded=False):
                    st.write(dflp[['BU','Prov','Kab_APBD','Efisiensi','omset']])
                # dflp = dflp.replace(to_replace=0,value=np.NAN)
                kolom = dfm.variable.unique().tolist()
                f1=dflp[s1]/dflp['Total_beban']
                f2=dflp[s2]/dflp['Total_beban']
                f3=dflp[s3]/dflp['Total_beban']
                fig = go.Figure()
                fig.add_trace(go.Box(y=f1,name=s1))
                fig.add_trace(go.Box(y=f2,name=s2))
                fig.add_trace(go.Box(y=f3,name=s3))
                fig.add_trace(go.Scatter(x=kolom, y=dfm['value'],mode='lines',name=umkm))
                fig.update_layout(width=900,height=600,title="Comparison with Frontier MSME")
                st.plotly_chart(fig)

                with st.beta_expander('Efficiency Analysis', expanded=False):
                    c1,c2,c3,c4 = st.beta_columns((2,1,2,2))
                    with c1:
                        # bv = dfs.iloc[0,3:7].astype('float').tolist()
                        st.text_input(label=s1+" (%)",value=int(dfs[s1].mean()*10000)/100.0)
                        st.text_input(label=s2+" (%)",value=int(dfs[s2].mean()*10000)/100.0)
                        st.text_input(label=s3+" (%)",value=int(dfs[s3].mean()*10000)/100.0)
                    with c2:
                        st.empty()
                    with c3:
                        #min value
                        v1min = st.number_input(label=s1,value=f1.quantile(0.25)*100.0,min_value=0.0, max_value=100.0, step=1.0)
                        v2min = st.number_input(label=s2,value=f2.quantile(0.25)*100.0,min_value=0.0, max_value=100.0, step=1.0)
                        v3min = st.number_input(label=s3,value=f3.quantile(0.25)*100.0,min_value=0.0, max_value=100.0, step=1.0)
                    with c4:
                        #max value
                        v1max = st.number_input(label=s1,value=f1.max()*100.0,min_value=0.0, max_value=100.0, step=1.0)
                        v2max = st.number_input(label=s2,value=f2.max()*100,min_value=0.0, max_value=100.0, step=1.0)
                        v3max = st.number_input(label=s3,value=f3.max()*100.0,min_value=0.0, max_value=100.0, step=1.0)

                model = pd.read_excel('model.xlsx')
                be = model['efisiensi'].tolist()
                bo = model['omset'].tolist()
                # Create the LP model
                prob = LpProblem(name="Allocation Optimization",sense=LpMaximize)
                # Initialize the decision variables
                v1 = LpVariable(name=s1, lowBound=0)
                v2 = LpVariable(name=s2, lowBound=0)
                v3 = LpVariable(name=s3, lowBound=0)
                # bo = [-332304.0175,1.05645607860482,1.15648620346764,1.14590795340159]
                # be = [0.846524,2.4740504346002E-11,-1.23825409764219E-12,2.85633816289292E-11]
                efscore= be[0]+v1*be[1]*dff['Total_beban'].mean()+v2*be[2]*dff['Total_beban'].mean()+v3*be[3]*dff['Total_beban'].mean()
                # grscore = bo[0]+v1*bo[1]+v2*bo[2]+v3*bo[3]
                # prob += grscore
                prob += efscore
                # prob += grscore
                # Add the constraints to the model
                prob += (v1+v2+v3 <=1, "full_constraint")
                prob += (v1*100 >= v1min, "v1min")
                prob += (v2*100 >= v2min, "v2min")
                prob += (v3*100 >= v3min, "v3min")
                prob += (v1*100 <= v1max, "v1max")
                prob += (v2*100 <= v2max, "v2max")
                prob += (v3*100 <= v3max, "v3max")
                prob += (efscore <=1, "maxEff")
                prob += (efscore >=0, "minEff")

                # Solve the problem
                st.write("Calculation for optimum allocation")
                if st.button("Click to Run"):
                    status = prob.solve()
                    p1 =  pulp.value(v1)
                    p2 =  pulp.value(v2)
                    p3 =  pulp.value(v3)
                    total = int((p1+p2+p3)*10000)/100
                    outls = [p1,p2,p3]
                    # st.subheader(outls)
                    h1,h2 = st.beta_columns((5,3))
                    
                    with h1:
                        fig1 = go.Figure()
                        fig1.add_trace(go.Bar(x=kolom, y=dfm['value'],name='Current Allocation'))
                        fig1.add_trace(go.Bar(x=kolom, y=outls,name='Recommendation'))
                        fig1.update_layout(width=700, height=600)
                        st.plotly_chart(fig1)
                    with h2:
                        growth= bo[0]+p1*bo[1]*dff['Total_beban'].mean()+p2*bo[2]*dff['Total_beban'].mean()+p3*bo[3]*dff['Total_beban'].mean()
                        efficiency= be[0]+p1*be[1]*dff['Total_beban'].mean()+p2*be[2]*dff['Total_beban'].mean()+p3*be[3]*dff['Total_beban'].mean()
                        st.markdown('')
                        st.markdown('')
                        st.markdown('')
                        # st.write(status)
                        fig3 = go.Figure()
                        fig3.add_trace(go.Indicator(
                                        mode = "number+delta",
                                        # value = status*100,
                                        value = int(growth*100)/100,
                                        # value = f'{int(growth):,d}',
                                        title = {"text": "Revenue Prediction:"},
                                        delta = {'reference': int(dff.omset.mean()*100)/100, 'relative': False},
                                        # delta = {'reference': f'{int(dff.omset.mean()):,d}', 'relative': False},
                                        # # domain = {'x': [0, 0.5], 'y': [0.6, 1]},
                                        domain = {'x': [0, 0.5], 'y': [0, 0.4]},
                                        ))
                        fig3.add_trace(go.Indicator(
                                        mode = "number+delta",
                                        value = status*100,
                                        # value = int(efficiency*10000)/100,
                                        title = {"text": "Efficiency Score (%):"},
                                        delta = {'reference': int(dff.Efisiensi.mean()*10000)/100, 'relative': False},
                                        # domain = {'x': [0, 0.5], 'y': [0, 0.4]},
                                        domain = {'x': [0, 0.5], 'y': [0.6, 1]},
                                        ))
                        # fig3.update_layout(width=200)
                        st.plotly_chart(fig3)
                    st.subheader(f'Total Allocated Cost: {int(total)}%')
                    with st.beta_expander("Reallocation gap",expanded=False):
                        excess = int(dff['Total_beban']) * (100-total)
                        st.subheader(f'Rp {int(excess):,d}')
                        # st.number_input(label=" ",value=excess,min_value=0.0, max_value=1000000000.0, step=10.0)
                
if __name__=='__main__':
    main()
