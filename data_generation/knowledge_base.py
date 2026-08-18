"""
Historical knowledge base used for temporal question generation.
"""


class KnowledgeBase:
    """Curated knowledge base of historical events, people, and organizations."""

    def __init__(self):
        self.events = []
        self.people = []
        self.organizations = []

    def load(self):
        """Load all knowledge base data."""
        print("Loading knowledge base...")
        self._load_events()
        self._load_people()
        self._load_organizations()
        print(f"Loaded: {len(self.events)} events, {len(self.people)} people, "
              f"{len(self.organizations)} organizations")

    def get_stats(self) -> dict:
        return {
            'events': len(self.events),
            'people': len(self.people),
            'organizations': len(self.organizations),
        }

    def _load_events(self):
        events_data = [
            {'name': 'World War I', 'year': 1914, 'end_year': 1918, 'location': 'Europe', 'casualties': 17000000, 'domain': 'military'},
            {'name': 'World War II', 'year': 1939, 'end_year': 1945, 'location': 'Global', 'casualties': 75000000, 'domain': 'military'},
            {'name': 'Korean War', 'year': 1950, 'end_year': 1953, 'location': 'Korea', 'casualties': 3000000, 'domain': 'military'},
            {'name': 'Vietnam War', 'year': 1955, 'end_year': 1975, 'location': 'Vietnam', 'casualties': 3800000, 'domain': 'military'},
            {'name': 'Sputnik Launch', 'year': 1957, 'location': 'Soviet Union', 'casualties': 0, 'domain': 'science'},
            {'name': 'Moon Landing', 'year': 1969, 'location': 'United States', 'casualties': 0, 'domain': 'science'},
            {'name': 'First Human in Space', 'year': 1961, 'location': 'Soviet Union', 'casualties': 0, 'domain': 'science'},
            {'name': '2004 Indian Ocean Tsunami', 'year': 2004, 'location': 'Indian Ocean', 'casualties': 280000, 'domain': 'disaster'},
            {'name': 'Hurricane Katrina', 'year': 2005, 'location': 'United States', 'casualties': 1800, 'domain': 'disaster'},
            {'name': 'Haiti Earthquake', 'year': 2010, 'location': 'Haiti', 'casualties': 316000, 'domain': 'disaster'},
            {'name': 'September 11 Attacks', 'year': 2001, 'location': 'United States', 'casualties': 3000, 'domain': 'terrorism'},
            {'name': 'COVID-19 Pandemic', 'year': 2020, 'location': 'Global', 'casualties': 7000000, 'domain': 'health'},
            {'name': 'Arab Spring', 'year': 2011, 'location': 'Middle East', 'casualties': 100000, 'domain': 'politics'},
            {'name': 'Internet Creation', 'year': 1989, 'location': 'Global', 'casualties': 0, 'domain': 'technology'},
            {'name': 'iPhone Launch', 'year': 2007, 'location': 'United States', 'casualties': 0, 'domain': 'technology'},
            {'name': 'Facebook Launch', 'year': 2004, 'location': 'United States', 'casualties': 0, 'domain': 'technology'},
        ]
        for i, d in enumerate(events_data):
            self.events.append({
                'id': f"EVENT_{i}",
                'name': d['name'],
                'year': d['year'],
                'end_year': d.get('end_year', d['year']),
                'location': d['location'],
                'casualties': d['casualties'],
                'domain': d['domain'],
                'source': 'curated',
            })

    def _load_people(self):
        people_data = [
            {'name': 'Albert Einstein', 'birth': 1879, 'death': 1955, 'country': 'Germany', 'field': 'Physics'},
            {'name': 'Marie Curie', 'birth': 1867, 'death': 1934, 'country': 'Poland', 'field': 'Chemistry'},
            {'name': 'Stephen Hawking', 'birth': 1942, 'death': 2018, 'country': 'United Kingdom', 'field': 'Physics'},
            {'name': 'Winston Churchill', 'birth': 1874, 'death': 1965, 'country': 'United Kingdom', 'field': 'Politics'},
            {'name': 'Nelson Mandela', 'birth': 1918, 'death': 2013, 'country': 'South Africa', 'field': 'Politics'},
            {'name': 'John F. Kennedy', 'birth': 1917, 'death': 1963, 'country': 'United States', 'field': 'Politics'},
            {'name': 'Steve Jobs', 'birth': 1955, 'death': 2011, 'country': 'United States', 'field': 'Technology'},
            {'name': 'Bill Gates', 'birth': 1955, 'death': None, 'country': 'United States', 'field': 'Technology'},
            {'name': 'Elon Musk', 'birth': 1971, 'death': None, 'country': 'United States', 'field': 'Technology'},
        ]
        for i, d in enumerate(people_data):
            self.people.append({
                'id': f"PERSON_{i}",
                'name': d['name'],
                'birth_year': d['birth'],
                'death_year': d['death'],
                'country': d['country'],
                'field': d['field'],
                'source': 'curated',
            })

    def _load_organizations(self):
        orgs_data = [
            {'name': 'United Nations', 'founded': 1945, 'country': 'International', 'type': 'International Organization'},
            {'name': 'NASA', 'founded': 1958, 'country': 'United States', 'type': 'Space Agency'},
            {'name': 'Apple Inc.', 'founded': 1976, 'country': 'United States', 'type': 'Technology Company'},
            {'name': 'Microsoft Corporation', 'founded': 1975, 'country': 'United States', 'type': 'Technology Company'},
            {'name': 'Google', 'founded': 1998, 'country': 'United States', 'type': 'Technology Company'},
        ]
        for i, d in enumerate(orgs_data):
            self.organizations.append({
                'id': f"ORG_{i}",
                'name': d['name'],
                'inception_year': d['founded'],
                'country': d['country'],
                'type': d['type'],
                'source': 'curated',
            })
