import requests
import pandas as pd

# URL for the endpoint
url = "http://127.0.0.1:8000/admin/stop_server"

# Send a POST request
response = requests.post(url)

# Handle the response
if response.status_code == 200:
    try:
        data = response.json()  #Parse the JSON response
        final_attendance = data.get("final_attendance", [])
        
        # Create a DataFrame and save it to an Excel file
        df = pd.DataFrame(final_attendance)
        output_path = 'final_attendance_from_site.xlsx'
        df.to_excel(output_path, index=False)
        print(f"Data saved to {output_path}")
    except ValueError:
        print("Response is not in JSON format. Response content:")
        print(response.text)
else:
    print(f"Failed to fetch data. HTTP Status Code: {response.status_code}")
    print(response.text)
