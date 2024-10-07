from openai import OpenAI
import os
client = OpenAI(api_key='sk-svcacct-WaBPRBqFS9ICIVNSIbKBoja6n5yfdtKtnkncf-xAh5koGdGeht7NsgSFWbRjKYmXpzvT3BlbkFJ4B4NorQxTMSdbjw4HKAaHpeKzv5t3Wgr_o8TA4N41Tc8tx2MkpG3Ah6o2ywvgULci7gA')

# Set OpenAI API key

# Define the feature and categories
feature_options = {
    'body_part': ['top', 'skirt suit', 'trousers', 'overcoat'],
    'accessories': [
        'No accessories', 'basic rope', 'button_belt_Ribbon', 'button_Metal_Three-dimensional decoration',
        'button', 'zipper_basic rope', 'belt_button_lace', 'button_rivet', 'button_basic rope', 'belt',
        'button_belt', 'lace', 'lace_Three-dimensional decoration', 'belt_zipper_sequins', 'belt_basic rope',
        'basic rope_belt', 'basic rope_zipper', 'Ribbon_zipper', 'belt_bead', 'zipper', 'button_Metal', 
        'lace_bead', 'bead', 'sequins_zipper', 'Ribbon', 'zipper_button', 'button_lace_zipper', 
        'button_zipper_Metal', 'Ribbon_button_Three-dimensional decoration', 'lace_zipper', 'button_Buckle', 
        'button_belt_zipper'
    ],
    'cloth_orientation': ['front', 'back'],
    'coat_length': ['conventional', 'high waist', 'middle length', 'knee high'],
    'category': [
        'Hoodie', 'leather skirt', 'leather pants', 'sling', 'casual pants', 'knitted sweater', 'shorts', 
        'polo shirt', 'dress', 'leather jacket', 'T-shirt', 'skirt', 'jacket', 'sweater', 'shirt', 
        'Chiffon shirt (lace shirt)', 'Wide leg pants', 'suit', 'jeans', 'Rompers', 'Woolen coat', 'down jacket', 
        'waistcoat', 'Coat (windbreaker)', 'bottoming shirt', 'trousers'
    ],
    'season': ['summer', 'winter', 'spring and autumn'],
    'sleeve_length': ['long', 'short', 'none'],
    'pocket': ['side pocket', 'no','patch pocket' ],
    'elbow': ['conventional','none'],  
    'color': [
        'Rose-pink cotton', 'Fantasia green', 'Black cotton-blend', 'Blue cotton', 
        'Prussian-blue/mint-green/purple Arjuna', 'Black wool', 'Green patterned', 
        'Army green', 'Orange/black/blue recycled', 'Light pink', 'Ash brown', 'Midnight black', 
        'Speed red', 'Your wardrobe', 'Black button-front', 
        'White lambskin/cashmere-wool', 'Crystal rose', 'Lime green', 'Jet-black quilted', 'Argento grey', 
        'Bordeaux red', 'Blake tie-waist', 'Green silk', 'Navy cotton', 'Khaki cotton', 'Mid-brown',
        'White cotton', 'Moss green', 'Baby-blue mohair-wool', 'Alexander McQueen', 'Brown cotton', 'Red', 
        'Sailor blue', 'Beige cashmere', 'Yellow virgin', 'Brown wool-linen', 'Coconut brown', 'Grass green', 
        'Sky blue/white', 'Soft pink', 'Black cotton', 'Sage blue', 'Light blue/white', 'Dark green', 
        'Blue wool', 'Black virgin', 'Blue/black/white cotton', 'Black knitted', 'Dark grey', 'Bordeaux stretch', 
        'Camel-brown/latte-beige cotton', 'Praline pink', 'Grey mohair-blend', 'Pale pink', 'Pink wool/silk', 
        'Ocean blue', 'Mustard yellow', 'Black/blue asymmetric', 'Blue organic', 'Green/white cotton', 
        'Pale grey', 'Dark blue', 'Beige cotton-wool', 'Orange/black cotton', 'Bubble pink', 
        'Salamander-red cotton', 'Tan virgin', 'Beige belted', 'Blue linen', 'Red triathlon', 
        'White cotton-linen', 'Light grey', 'Red logo', 'Taupe brown', 'Cayenne red', 'Blue silk', 'Caramel brown', 
        'Grey cotton', 'Experimentation, innovation', 'Orange cotton', 'Red leather-effect', 
        'Light blue', 'Bright red', 'Orange sorbet','Black viscose-blend', 'Red feather', 'Cognac cotton', 
        'Grey cotton-cashmere', 'Navy blue', 'Olive green', 'Mid grey', 'When life', 'Beige cashmere-virgin', 
        'Bordeaux red/multicolour', 'Green wool', 'Still wondering', 'Cyan green', 'White sleeveless', 
        'Slate blue', 'khaki green','Black silk', 'Rose-pink', 'cotton-blend', 'TOM OF', 
        'Black/fuchsia VLTN', 'Simple lines', 'White virgin', 'Medium blue', 'Fire-orange/black wool-blend', 
        'Brick red', 'Pink Lannic', 'Banana-yellow/white cotton', 'Water green', 'Light indigo', 
        'Grey/white cotton-cashmere', 'Black/white silk/wool', 'White cotton/silk', 'Dark night', 
        'Grey wool-blend', 'Sage green', 'Cappucino cashmere', 'Acid-green cotton', 'Khaki green', 
        'Anthracite-grey cashmere', 'Black tailored', 'Pink silk', 'Coffee grey', 'Black linen', 'Black Medusa', 
        'Sandshell beige', 'Beige stretch', 'Navy linen-cotton', 'Rinse wash', 'Blue/red stretch-cotton','Bright white','Purple','mud green'
    ],
     'collar': [
        'regular turtleneck', 'bandeau collar', 'none', 'round neck', 'Deep V neck', 'polo collar', 
        'shirt collar', 'Polo collar', 'regular high collar', 'V-neck', 'hooded', 'suit collar', 
        'Doll collar', 'square collar', 'half open collar', 'pilot collar'
    ],
    'craft': [
        'printing', 'no craft', 'other craft', 'patch_printing', 'patch', 'bow tie', 'Beaded rivets_Raw edge', 
        'patch_fine needle (thin thread)', 'bow tie_Lace up', 'printing_fine needle (thin thread)', 'Beaded rivets', 
        'Lace up', 'fine needle (thin thread)', 'Wood ear_fine needle (thin thread)_Beaded rivets', 'ruffles', 
        'bow tie_Beaded rivets', 'Beaded rivets_washed', 'printing and dyeing', 'solicit', 'washed', 
        'printing_printing and dyeing', 'Quilting_Lace up', 'washed_Raw edge_Beaded rivets', 'printing_Lace up', 
        'Quilting', 'patch_Quilting', 'solicit_Lace up', 'wear holes', 'solicit_ruffles_Wood ear', 'patch_ruffles', 
        'fine needle (thin thread)_patch', 'Lace up_patch', 'Pleating', 'shirring', 'Raw edge', 'ruffles_bow tie', 
        'washed_Beaded rivets', 'printing_bow tie', 'printing and dyeing_Quilting', 'Pleating_printing and dyeing'
    ],
    'style' : [
    'casual_simple_wild', 'casual_wild_simple', 'sports_Korean version_fresh', 'casual_simple', 
    'simple_sweet_lady', 'wild_commute_simple', 'simple_casual_commute', 'casual_sports_lady', 'simple_ladies', 
    'commute_casual_simple', 'simple_wild_neutral', 'simple_wild_casual', 'casual_simple_ladies', 
    'casual_wild_Korean version', 'simple_sweet_sports', 'simple_wild', 'wild_simple_casual', 
    'simple_wild_Korean version', 'simple_casual_wild', 'lady_commute_simple', 'simple_casual_Korean version', 
    'casual_fresh', 'ladies_fresh_lady', 'casual_neutral_simple', 'lady_ladies', 'sweet_Korean version_Ruili', 
    'casual_simple_neutral', 'simple_neutral', 'simple_Korean version_wild', 'casual_simple_commute', 
    'casual_simple_Korean version', 'casual_commute_simple', 'casual_simple_sweet', 'commute', 
    'lady_wild_Korean version', 'simple_commute_casual', 'commute_OL_simple', 'lady_ladies_simple', 
    'Korean version_casual_wild', 'lady_sweet', 'wild_casual_simple', 'casual_sports_wild', 
    'casual_sports_simple', 'casual_wild_neutral', 'casual_wild', 'wild_Ruili_commute', 'commute_Ruili_vintage', 
    'lady_Ruili_ladies', 'wild_Korean version_simple', 'simple_casual_fresh', 'lady_simple_commute', 
    'casual_simple_Ruili', 'lady_Korean version_ladies', 'casual_neutral_sports', 'lady_casual_simple', 
    'simple_sweet_wild', 'simple_casual', 'sports_wild_casual', 'simple', 'casual_street_simple', 
    'simple_Korean version_casual', 'casual_neutral_wild', 'wild_commute_lady', 'simple_lady_ladies', 
    'simple_casual_sports', 'sweet_vintage_Mori Department', 'sweet_lady_Mori Department', 'casual_street', 
    'simple_lady_casual', 'lady_OL_sweet', 'simple_wild_sports', 'casual_punk_neutral', 'simple_casual_neutral', 
    'casual_simple_sports', 'casual_commute_street', 'simple_OL_commute', 'simple_wild_sweet', 
    'simple_commute_wild', 'sweet_lady_commute', 'simple_casual_OL', 'simple_commute', 'casual_wild_lady', 
    'casual_sweet_fresh', 'casual_wild_street', 'commute_simple_casual', 'wild_simple_sweet', 'lady_sweet_Ruili', 
    'simple_wild_lady', 'simple_OL_wild', 'casual_sports', 'Korean version_simple_casual', 
    'Korean version_ladies_commute', 'casual_sports_commute', 'wild_sweet_simple'
     ],
    'type_version' : ['fit', 'Slim fit', 'super loose', 'tight', 'loose'],
    'suit' : ['none', 'pants suit', 'sports suit', 'skirt suit', 'suit'],
    'suitable_age' : ['Children', 'youth', 'middle age'],

}
def extract_value_for_feature(user_input, feature):
    """
    Extract the value for a given feature from the user's input.
    
    @param user_input: The user's input string.
    @param feature: The feature we are extracting the value for.
    @return: The extracted value for the feature, if found; otherwise, None.
    """
    # Example basic extraction logic (you can customize it based on how the input is structured)
    for value in feature_options.get(feature, []):
        if value.lower() in user_input.lower():
            return value
    return None
# construct active prompt
# Function to construct active prompt and parse user input
def parse_user_input(user_input):
    """
    Constructs an active prompt for the OpenAI GPT model to parse user input and extract only the mentioned clothing features.
    
    @param user_input: A string containing the user's input/request.
    @return: The GPT-generated response with extracted features in a dictionary format.
    """
    # Initialize an empty list to store only the valid feature-related prompts
    relevant_features = []

    # Iterate through each feature and its allowed values from the feature_options dictionary
    for feature, values in feature_options.items():
        # Attempt to extract the value of the feature from the user input
        extracted_value = extract_value_for_feature(user_input, feature)
        
        # If a valid value is extracted, add it to the relevant features
        if extracted_value:
            if isinstance(values, list):
                # Add a string describing the allowed values for the feature
                relevant_features.append(f"{feature}: {extracted_value}")
    
    # Join the relevant features into a single string to guide the GPT model
    options_prompt = ", ".join(relevant_features)

    # Enhanced prompt engineering to instruct GPT to extract only mentioned features
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            # System role: provide clear instructions to GPT to only extract mentioned features
            {"role": "system", "content": f"You are a clothing recommendation assistant. Your task is to extract features like color, body_part, season, and other attributes that are **explicitly mentioned** in user requests for clothes. Ignore any features that the user does not mention. The values of the features must be consistent with the following list: {options_prompt}. Ensure the output is a structured dictionary and do not include any feature that was not mentioned by the user."},

            # User input prompt asking GPT to extract only relevant features mentioned in the user's request
            {"role": "user", "content": f"User input: '{user_input}' Please extract the mentioned attributes in dictionary format, ignoring anything not explicitly mentioned by the user. The list of possible features is: {list(feature_options.keys())}"}
        ]
    )

    # Return GPT's response, which contains the parsed features as a dictionary
    return response.choices[0].message.content



# 示例：用户输入
user_input = "I want a blue long-sleeve shirt without pockets for summer."
parsed_result = parse_user_input(user_input)
print(parsed_result)
