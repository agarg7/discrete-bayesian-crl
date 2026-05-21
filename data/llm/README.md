# LLM data

This provides additional details on the LLM-based data generation pipeline, including the prompts used at each stage and sample outputs. All responses are generating using gpt-4o.

- `data_85-15.json` contains the the dataset the model is fit to with stochastic interventions 
- `data_100-0.json` contains interventional profiles with hard interventions 

### Environment_list:
1. Democrat
2. Republican
3. Supports Free Trade
4. Against Free Trade
5. Supports Government Regulation
6. Against Government Regulation
7. Believes Benefits Are Inadequate
8. Believes Benefits Are Adequate
9. Pro-Immigration
10. Anti-Immigration
11. Basline

### Question_list
```
T1: When the US expands free trade, the trade-offs are worth it.
T2: More and cheaper goods are WORTH having less quality control in production.
T3: More and cheaper goods are WORTH the increased stress on the borders.
T4: More and cheaper goods are WORTH the increased competition for domestic goods.
T5: Creating different or new jobs is WORTH eliminating some existing jobs.
T6: A bigger market for US exporters is WORTH increased competition for US domestic producers.
R1: In general, government regulations are NECESSARY to protect the public interest.
R2: Regulating environmental pollution protects the public interest MORE than it harms businesses.
R3: Regulating workplace safety protects the public interest MORE than it harms businesses.
R4: Regulations of banks and financial products are NECESSARY to protect the public interest.
R5: Regulations of product safety are NECESSARY to protect the public interest.
R6: Regulating food and drug safety protects the public interest MORE than it harms businesses.
P1: Poor people today have it hard, without enough government benefits to succeed.
P2: Poor people today have it hard, without enough government benefits to attain education.
P3: Poor people today have it hard, without enough government benefits to attain shelter.
P4: Poor people today have it hard, without enough government benefits to attain healthcare.
P5: Poor people today have it hard, without enough government benefits for job training.
P6: Poor people today have it hard, without enough government benefits to attain food.
I1: Recent immigrants STRENGTHEN our country.
I2: Recent immigrants STRENGTHEN the US job market.
I3: Recent immigrants ENRICH American cultural life.
I4: Recent legal immigrants STRENGTHEN US national security.
I5: Recent undocumented/illegal immigrants STRENGTHEN US national security.
I6: The US–Mexico border is sufficiently secure.
G1: What is your political party affiliation? (1-100 Scale: 1 = Strong Republican, 100 = Strong Democrat)
G2: How do you feel about the job Donald Trump is doing as president? (1-100 Scale: 1 = strongly approve, 100 = strongly disapprove)
G3: Who will you vote for in the House of Representatives in 2018? (1-100 Scale: 1 = Definitely Republican, 100 = Definitely Democrat)"]}
```


---

## (i) Sample demographic data from real survey respondents (PredictWise Data)

### Demographic data used:

```
gender, age, marital status, number of children, education, employment status, career, race, income, spoken languages, organizational role, number of employees, country, area, city, TSmart urbanicity, voter-base party, voter-base voter score, TSmart presidential general turnout score
```

## (ii) Extrapolate demographic data into detailed respondent profiles (LLM)

### System prompt:

```
You are a helpful assistant generating realistic survey respondent profiles. 
```
### User Prompt:

```
Below is a demographic profile for a random survey respondent: 

{profile data from (i)}

Generate a representative fictional survey respondent profile for this demographic profile. 
Include demographic variables (e.g., age, race, education, income, region), political affiliation, and detailed views on the following four issues: trade, regulation, poverty, and immigration. 
For each belief, provide a short explanation of *why* this person holds that belief.
```

### Sample Output: 

```
**Respondent Profile:**

- **Name:** John Smith
- **Gender:** Male
- **Age:** 72
- **Marital Status:** Married
- **Number Of Children:** None
- **Education:** Vocational Technical College
- **Employment Status:** Retired
- **Career:** Former Dental Technician
- **Race:** White
- **Income:** Lower Middle-Class
- **Country:** US
- **Area:** Wisconsin
- **City:** Green Bay
- **Political Affiliation:** Moderate Democrat

**Views on Key Issues:**

1. **Trade:**
   - **Belief:** John supports fair trade policies that protect American jobs while allowing some degree of globalization.
   - **Explanation:** Having grown up in a manufacturing town, John witnessed firsthand the impact that outsourcing and international competition had on local employment. He believes in fair trade that doesn't disadvantage American workers.

2. **Regulation:**
   - **Belief:** He favors moderate regulation, particularly in healthcare and environmental standards.
   - **Explanation:** As a retired dental technician, John values the importance of health regulations to ensure safety and efficacy. Additionally, environmental issues are important to him because of Wisconsin's natural beauty, which he wants to preserve for future generations.

3. **Poverty:**
   - **Belief:** John believes in strengthening social safety nets to support those in poverty.
   - **Explanation:** Through his church, John often volunteers at local food banks and has seen the struggles of people living below the poverty line. This personal involvement has instilled a belief in stronger governmental support for the needy.

4. **Immigration:**
   - **Belief:** He supports a compassionate approach to immigration reform that includes a pathway to citizenship for undocumented immigrants already contributing to society.
   - **Explanation:** John's community has become more diverse over the years, and he has come to appreciate the cultural and economic contributions of immigrants in Green Bay. He believes in balanced reforms that offer opportunities for those seeking a better life.John's views reflect his moderate Democratic stance, influenced by his personal experiences and community environment in Green Bay, Wisconsin.
```



## (iii) Apply causal interventions to respondent profiles (LLM)

### System prompt:

```
You are going to be given a demographic profile. Modify the person's profile with the following causal intervention: 

Set {issue opinion} to {specified side}

Update their **causally downstream** beliefs and reasoning accordingly. Return a fully rewritten profile in the same format and tone.
```

### User Prompt:

```
{profile text from (ii)}
```

### Sample Output:

```
**Respondent Profile:**

- **Name:** John Smith
- **Gender:** Male
- **Age:** 72
- **Marital Status:** Married
- **Number Of Children:** None
- **Education:** Vocational Technical College
- **Employment Status:** Retired
- **Career:** Former Dental Technician
- **Race:** White
- **Income:** Lower Middle-Class
- **Country:** US
- **Area:** Wisconsin
- **City:** Green Bay
- **Political Affiliation:** Republican

**Views on Key Issues:**

1. **Trade:**
   - **Belief:** John supports free trade policies that bolster American economic interests and prioritize national prosperity.
   - **Explanation:** Having witnessed the effects of global competition on local manufacturing, John believes in policies that enhance U.S. competitiveness. He supports strategies that encourage American businesses to thrive in a global market.

2. **Regulation:**
   - **Belief:** He favors reducing regulation to promote business growth while still maintaining essential health and environmental standards.
   - **Explanation:** As a former dental technician, John understands the importance of sensible health regulations but advocates for minimizing governmental constraints that can hinder economic development, aiming for a balance that preserves Wisconsin’s natural beauty without stifling industry.

3. **Poverty:**
   - **Belief:** John believes in empowering communities through local initiatives and private charities rather than expanding governmental programs.
   - **Explanation:** With experience volunteering at food banks, he has witnessed the positive impact of community-driven efforts. John supports limited governmental roles in favor of initiatives that encourage self-sufficiency and community support.

4. **Immigration:**
   - **Belief:** He supports strong border security and advocates for a merit-based immigration system.
   - **Explanation:** While appreciative of Green Bay's diversity, John believes in prioritizing national security and economic considerations. He supports reforms that ensure immigrants can contribute effectively to American society.

John's views reflect his Republican stance, shaped by his personal experiences and a focus on economic growth, community empowerment, and national security in Green Bay, Wisconsin.
```




## (iv) Generate full survey responses based on intervened profiles (LLM)

### System prompt:

```
You are a survey respondent in 2018. Your identity is defined strictly by the following profile:

{profile text from (iii)}

You must embody a person with these attributes in your answers. You are not allowed to reinterpret, ignore, or randomly change them. All your answers should reflect the specified background and views.
```

### User Prompt:

```
Respond to these questions on a 1–100 scale (1 = strongly disagree, 100 = strongly agree):
    Trade:  
        1. When the US expands free trade, the trade-offs are worth it.
        2. More and cheaper goods are WORTH having less quality control in production.
        3. More and cheaper goods are WORTH the increased stress on the borders.
        4. More and cheaper goods are WORTH the increased competition for domestic goods.
        5. Creating different or new jobs is WORTH eliminating some existing jobs.
        6. A bigger market for US exporters is WORTH increased competition for US domestic producers.

    Regulation:
        1. In general, government regulations are NECESSARY to protect the public interest.
        2. Regulating environmental pollution protects the public interest MORE than it harms businesses.
        3. Regulating workplace safety protects the public interest MORE than it harms businesses.
        4. Regulations of banks and financial products are NECESSARY to protect the public interest.
        5. Regulations of product safety are NECESSARY to protect the public interest.
        6. Regulating food and drug safety protects the public interest MORE than it harms businesses.
    
    Poverty:
        1. Poor people today have it hard, without enough government benefits to succeed.
        2. Poor people today have it hard, without enough government benefits to attain education.
        3. Poor people today have it hard, without enough government benefits to attain shelter.
        4. Poor people today have it hard, without enough government benefits to attain healthcare.
        5. Poor people today have it hard, without enough government benefits for job training.
        6. Poor people today have it hard, without enough government benefits to attain food.
    
    Immigration:
        1. Recent immigrants STRENGTHEN our country.
        2. Recent immigrants STRENGTHEN the US job market.
        3. Recent immigrants ENRICH American cultural life.
        4. Recent legal immigrants STRENGTHEN US national security.
        5. Recent undocumented/illegal immigrants STRENGTHEN US national security.
        6. The US–Mexico border is sufficiently secure.
    
    General:
        1. What is your political party affiliation? (1-100 Scale: 1 = Strong Republican, 100 = Strong Democrat)
        2. How do you feel about the job Donald Trump is doing as president? (1-100 Scale: 1 = strongly approve, 100 = strongly disapprove)
        3. Who will you vote for in the House of Representatives in 2018? (1-100 Scale: 1 = Definitely Republican, 100 = Definitely Democrat)
```
### JSON Schema:

```json
"type": "object",
"properties": {
  "Trade": {
    "type": "object",
    "properties": {
      "T1": { "type": "integer", "minimum": 1, "maximum": 100, "description": "When the US expands free trade, the trade-offs are worth it." },
      [...]
    },
    "required": ["T1", "T2", "T3", "T4", "T5", "T6"],
    "additionalProperties": False
  },
  "Regulation": {  [...] },
  "Poverty": { [...] },
  "Immigration": {  [...] },
  "General": {  [...] }
},
"required": ["Trade", "Regulation", "Poverty", "Immigration", "General"],
"additionalProperties": False
```

### Sample Output:

```json
"Trade": {"T3": 45, "T5": 65, "T1": 75, "T6": 70, "T2": 50, "T4": 55},
"General": {"G3": 25, "G1": 20, "G2": 30},
"Immigration": {"I5": 30,"I1": 55,"I2": 50, "I6": 40,"I3": 60,"I4": 50},
"Regulation": {"R4": 65, "R3": 75, "R6": 80, "R1": 60, "R5": 85, "R2": 70},
"Poverty": {"P5": 50, "P2": 45, "P1": 40, "P4": 55, "P3": 50, "P6": 45} 
```

---

