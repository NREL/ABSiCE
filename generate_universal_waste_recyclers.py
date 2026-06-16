"""
Script to generate Universal_Waste_Recyclers_data.csv from DTSC Universal Waste Recyclers
"""
import pandas as pd
import requests
from bs4 import BeautifulSoup
import geopy
from geopy.geocoders import Nominatim, ArcGIS
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time
import re
import ssl
import certifi

def get_facilities_from_dtsc():
    """Scrape facility data from DTSC website"""
    url = "https://dtsc.ca.gov/list-of-universal-waste-recyclers-that-treat-pv-modules/"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all table rows in the page
        facilities = []
        tables = soup.find_all('table')
        
        for table in tables:
            rows = table.find_all('tr')
            for row in rows[1:]:  # Skip header row
                cols = row.find_all('td')
                if len(cols) >= 2:
                    facility_name = cols[0].get_text(strip=True)
                    address = cols[1].get_text(strip=True)
                    
                    # Skip empty names
                    if not facility_name:
                        continue
                        
                    facilities.append({
                        'name': facility_name,
                        'address': address
                    })
        
        return facilities
    
    except Exception as e:
        print(f"Error fetching data from DTSC website: {e}")
        return []

def get_facilities_from_csv(csv_filename: str, facility_name_col: str = 'Facility Name', address_col: str = 'Address'):
    """Read facility data from CSV file
    
    Args:
        csv_filename: Path to CSV file containing facility information
        
    Returns:
        List of dictionaries with 'name' and 'address' keys
    """
    try:
        df = pd.read_csv(csv_filename)
        
        facilities = []
        for _, row in df.iterrows():
            # Assuming CSV has columns for facility name and address
            # Adjust column names as needed based on actual CSV structure
            facility_name = row.get(facility_name_col)
            address = row.get(address_col)

            if address is None or pd.isna(address) or facility_name is None or pd.isna(facility_name):
                raise ValueError(f"column {facility_name_col} or {address_col} is missing in the CSV file.")
            
            # Skip empty names
            if not facility_name or pd.isna(facility_name):
                continue
                
            facilities.append({
                'name': str(facility_name).strip(),
                'address': str(address).strip()
            })
        
        return facilities
    
    except Exception as e:
        print(f"Error reading CSV file {csv_filename}: {e}")
        return []

def get_coordinates(address, geolocator, max_retries=3):
    """Get latitude and longitude from address using geopy"""
    for attempt in range(max_retries):
        try:
            time.sleep(1)  # Rate limiting
            location = geolocator.geocode(address, timeout=10)
            if location:
                return location.latitude, location.longitude
            else:
                print(f"  Could not geocode: {address}")
                return None, None
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            print(f"  Geocoding error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return None, None
            time.sleep(2 ** attempt)  # Exponential backoff
    return None, None
    
ssl._create_default_https_context = ssl._create_unverified_context

def get_state_and_city_from_address(address):
    """Extract state and city from address using string parsing
        Example format: "123 Main St, Suite 100, Springfield, IL"
    """
    address_parts = [part.strip() for part in address.split(',')]
    if len(address_parts) < 3:
        return None, None
    city = address_parts[-2]
    state_zip = address_parts[-1].split()
    state = state_zip[0] if len(state_zip) > 0 else None
    return state, city

def main():
    print("Starting Universal Waste Recyclers data generation...")
    
    # Get facilities from DTSC website
    # print("\n1. Fetching facilities from DTSC website...")
    # facilities = get_facilities_from_dtsc()
    print("\n1. Fetching facilities from CSV file...")
    facilities = get_facilities_from_csv(
        csv_filename='/Users/pghosh/SOLAR/Universal_Waste_Recyclers.csv', 
        facility_name_col='Name of UW Recycler', 
        address_col="Recycler's Address"
    )
    print(f"   Found {len(facilities)} facilities (after filtering out recycling facilities)")
    
    if not facilities:
        print("No facilities found. Exiting.")
        return
        
    # Initialize geolocator
    print("\n3. Geocoding addresses...")
    geolocator = ArcGIS()
    
    # Process each facility
    results = []
    for i, facility in enumerate(facilities):
        print(f"   Processing {i+1}/{len(facilities)}: {facility['name']}")
        
        lat, lon = get_coordinates(facility['address'], geolocator)
        state, city = get_state_and_city_from_address(facility['address'])
        # make sure state is in abbreviation format
        if state and len(state) > 2:
            state = geopy.country_names.get(state, state)
        
        assert state is not None, "State could not be determined"
        assert city is not None, "City could not be determined"
        
        if lat is not None and lon is not None:
            results.append({
                'Recycler Name': facility['name'],
                'State': state,
                'City': city,
                'Longitude': lon,
                'Latitude': lat,
                'RCRA permit': False,
                'Universal Waste Permit': True
            })
        else:
            print(f"   Skipping due to geocoding failure")
    
    # Create DataFrame and save to CSV
    print(f"\n4. Creating CSV file with {len(results)} facilities...")
    df = pd.DataFrame(results)
    
    output_path = '/Users/pghosh/SOLAR/ABSiCE/TEMP/Universal_Waste_Recyclers_data.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Successfully created {output_path}")
    print(f"  Total facilities: {len(df)}")
    print("\nSample of generated data:")
    print(df.head())

if __name__ == "__main__":
    main()
