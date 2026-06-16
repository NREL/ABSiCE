"""
Script to generate Universal_Waste_Landfills_data.csv from DTSC Universal Waste Handlers
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
    url = "https://dtsc.ca.gov/photovoltaic-modules-pv-modules-universal-waste-management-regulations_uw-handlers/"
    
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
                    
                    # Skip if name contains "recycling" (case insensitive)
                    if 'recycling' in facility_name.lower():
                        continue
                    
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
            
            # Skip if name contains "recycling" (case insensitive)
            if 'recycling' in str(facility_name).lower():
                continue
            
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

def calculate_ca_average_cost():
    """Calculate average $/ton for CA facilities from existing data"""
    # add state and csv file as arguments
    try:
        df = pd.read_csv('/Users/pghosh/SOLAR/ABSiCE/TEMP/Landfills_data.csv', index_col=0)
        
        # Filter for CA facilities
        ca_facilities = df[df['Facility Name'].str.contains(' CA', case=False, na=False)]
        
        if len(ca_facilities) == 0:
            print("No CA facilities found in existing data, using default value of 175")
            return 175
        
        mean_ca_cost = ca_facilities['$/ Ton'].mean()
        print(f"Mean $/ton for CA facilities: {mean_ca_cost:.2f}")
        
        # Average of mean CA cost and 175 (take from csv file)
        avg_cost = (mean_ca_cost + 175) / 2
        print(f"Calculated average cost: {avg_cost:.2f}")
        
        return avg_cost
        
    except Exception as e:
        print(f"Error calculating CA average cost: {e}")
        return 175  # Default fallback
    
ssl._create_default_https_context = ssl._create_unverified_context

def main():
    print("Starting Universal Waste Landfills data generation...")
    
    # Get facilities from DTSC website
    # print("\n1. Fetching facilities from DTSC website...")
    # facilities = get_facilities_from_dtsc()
    print("\n1. Fetching facilities from CSV file...")
    facilities = get_facilities_from_csv(
        csv_filename='/Users/pghosh/SOLAR/Universal_Waste_Handlers.csv', 
        facility_name_col='Name of UW Handler', 
        address_col="Handler's Street Address"
    )
    print(f"   Found {len(facilities)} facilities (after filtering out recycling facilities)")
    
    if not facilities:
        print("No facilities found. Exiting.")
        return
    
    # Calculate average cost
    print("\n2. Calculating average $/ton cost...")
    avg_cost = calculate_ca_average_cost()
    
    # Initialize geolocator
    print("\n3. Geocoding addresses...")
    # Create a default SSL context using the certifi CA bundle
    # ctx = ssl.create_default_context(cafile=certifi.where())
    geolocator =ArcGIS()
    
    # Process each facility
    results = []
    for i, facility in enumerate(facilities):
        print(f"   Processing {i+1}/{len(facilities)}: {facility['name']}")
        
        lat, lon = get_coordinates(facility['address'], geolocator)
        
        if lat is not None and lon is not None:
            results.append({
                'Facility Name': facility['name'],
                'Longitude': lon,
                'Latitude': lat,
                '$/ Ton': avg_cost
            })
        else:
            print(f"   Skipping due to geocoding failure")
    
    # Create DataFrame and save to CSV
    print(f"\n4. Creating CSV file with {len(results)} facilities...")
    df = pd.DataFrame(results)
    
    output_path = '/Users/pghosh/SOLAR/ABSiCE/TEMP/Universal_Waste_Landfills_data.csv'
    df.to_csv(output_path)
    
    print(f"\n✓ Successfully created {output_path}")
    print(f"  Total facilities: {len(df)}")
    print(f"  $/Ton value: {avg_cost:.2f}")
    print("\nSample of generated data:")
    print(df.head())

if __name__ == "__main__":
    main()
