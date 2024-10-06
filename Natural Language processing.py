pip install openai googlemaps
import openai
import googlemaps
import json
import requests
openai.api_key = 'your_openai_api_key'
gmaps = googlemaps.Client(key='your_google_maps_api_key')
def get_species_location(species_name):
    # Get place information for the species location
    result = gmaps.places(query=species_name)
    if result['status'] == 'OK':
        location = result['results'][0]['geometry']['location']
        address = result['results'][0]['formatted_address']
        print(f"Location for {species_name}: {address} at {location}")
        return location, address
    else:
        print(f"Could not find location for {species_name}.")
        return None, None
def classify_and_describe_species(species_name, location_data):
    # Create a query for the model
    prompt = f"""
    A new species has been found named '{species_name}'. The species is located at:
    Latitude: {location_data[0]['lat']}
    Longitude: {location_data[0]['lng']}
    Based on the habitat and location, classify if the species is likely 'native' or 'invasive'.
    
    Also, provide a brief description with key characteristics, any known environmental impact, and conservation notes if invasive.
    """
    # Call OpenAI API
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.6,
        max_tokens=300
    )
    
    classification = response.choices[0].text.strip()
    print("Classification and Details:\n", classification)
    return classification
def main(species_name):
    location_data, address = get_species_location(species_name)
    if location_data:
        result = classify_and_describe_species(species_name, location_data)
        print(f"Species Name: {species_name}\nAddress: {address}\nClassification:\n{result}")
    else:
        print("Species location could not be retrieved.")
        
# Example usage
main("Lionfish")
 make the code stilll big in single prompt 




