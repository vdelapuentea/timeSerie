from google.cloud import storage
from google.cloud import bigquery
from google.oauth2 import service_account
def load_data_sniim():
    client = bigquery.Client()
    data_sniim_estado = client.query(f""" SELECT DATE(date_time) AS Fecha, Estado, AVG(valor) AS Valor_Promedio_SNIIM FROM (
      SELECT A.*, B.Estado
      FROM ( SELECT * FROM `timeseries-387916.SNIIM.scraping` ) A 
      LEFT JOIN ( SELECT * FROM `timeseries-387916.SNIIM.grupos` ) B
      ON A.Centro_distribucion = B.Ingenio)
    GROUP BY Fecha, Estado;""").to_dataframe() 
    return data_sniim_estado
