"""Apply prototype rules and indicative keyword checks to uploaded documents."""

import re
import unicodedata

class DocumentAgent:
    """Analyzes policy documents and conditions"""

    def __init__(self):
        print("[OK] Document Agent initialized")

    def process(
        self,
        application,
        risk_result,
        document_text=None,
        document_name=None,
        document_provided=None,
        document_size_bytes=None,
        document_status=None,
        document_bundle=None,
    ):
        """Analyze the submitted application and optionally inspect extracted text.

        When ``document_bundle`` is provided (multi-document intake), it takes
        precedence over the single-document arguments.
        """

        if document_bundle is not None:
            document_text = document_bundle.get('combined_text') or None
            document_name = document_bundle.get('filenames_label')
            document_provided = bool(document_bundle.get('documents'))
            document_size_bytes = document_bundle.get('total_size_bytes')
            document_status = document_bundle.get('aggregate_status')

        result = {
            'agent': 'Document',
            'document_checks': [],
            'exclusions': [],
            'conditions': [],
            'warnings': [],
            'documents': list(document_bundle.get('documents', [])) if document_bundle else [],
            'document_checklist': document_bundle.get('checklist') if document_bundle else None,
            'uploaded_document': {
                'provided': bool(document_text) if document_provided is None else document_provided,
                'filename': document_name,
                'size_bytes': document_size_bytes,
                'characters_extracted': len(document_text) if document_text else 0,
                'extraction_status': document_status or (
                    'text_extracted' if document_text else 'not_provided'
                ),
                'possible_terms_for_human_review': [],
                'affects_decision': False,
                'affects_risk_score': False,
                'scope_status': 'not_provided',
                'scope_reason': None,
            },
        }

        uploaded_document = result['uploaded_document']
        if uploaded_document['provided']:
            scope_status, scope_reason = self._classify_document_scope(
                document_name,
                document_text,
            )
            uploaded_document['scope_status'] = scope_status
            uploaded_document['scope_reason'] = scope_reason
            uploaded_document['affects_decision'] = scope_status != 'supported'
            if scope_status != 'supported':
                result['warnings'].append(scope_reason)

        if document_text and uploaded_document['scope_status'] == 'supported':
            review_terms = {
                'Fraude ou déclaration frauduleuse': r'\b(fraud|fraudulent|fraude|frauduleux|frauduleuse)\b',
                'Exclusion de garantie': r'\b(exclusion|exclu|exclue|excluded)\b',
                'Permis de conduire': r'\b(permis|driver.?s license|unlicensed)\b',
                'Usage commercial': r'\b(commercial|usage professionnel)\b',
                'Délai de déclaration': r'\b(30 jours|thirty days|délai de déclaration)\b',
                'Modification du véhicule': r'\b(modification|modifi[ée]s?)\b',
                'Compétition automobile': r'\b(racing|course automobile|compétition)\b',
            }
            lowered_document = document_text.casefold()
            result['uploaded_document']['possible_terms_for_human_review'] = [
                label
                for label, pattern in review_terms.items()
                if re.search(pattern, lowered_document)
            ]
            result['warnings'].append(
                'Repérage lexical indicatif uniquement : aucun sens contractuel ni effet sur la recommandation n’est établi. Une revue humaine du document original est requise.'
            )
        elif uploaded_document['provided'] and not document_text:
            result['warnings'].append(
                'La pièce est jointe au dossier, mais son contenu n’a pas pu être extrait et n’a pas été analysé.'
            )

        checklist = result.get('document_checklist')
        if checklist:
            unreadable = [
                document['filename']
                for document in result['documents']
                if document.get('extraction_status') not in {
                    'text_extracted',
                    'ocr_extracted',
                }
            ]
            if unreadable:
                uploaded_document['affects_decision'] = True
                result['warnings'].append(
                    'Pièce(s) non exploitable(s) : '
                    + ', '.join(unreadable)
                    + '. Une revue humaine de l’original est requise.'
                )
            missing = checklist.get('missing_mandatory') or []
            if missing:
                uploaded_document['affects_decision'] = True
                result['warnings'].append(
                    'Pièces obligatoires manquantes pour ce produit : '
                    + ', '.join(missing)
                    + '. Le dossier doit être escaladé (demande d’information / revue humaine).'
                )

        return result

    @staticmethod
    def _normalize_text(value):
        normalized = unicodedata.normalize('NFKD', value).casefold()
        without_accents = ''.join(
            character
            for character in normalized
            if not unicodedata.combining(character)
        )
        return re.sub(r'[^a-z0-9]+', ' ', without_accents).strip()

    def _classify_document_scope(self, document_name, document_text):
        """Fail closed unless the attachment contains recognizable auto-insurance context."""
        normalized_name = self._normalize_text(document_name or '')
        normalized_text = self._normalize_text(document_text or '')

        health_document_patterns = (
            r'\bmutuelle\b',
            r'\bassurance maladie\b',
            r'\bassurance sante\b',
            r'\bcomplementaire sante\b',
            r'\bremboursement(?:s)? de soins\b',
            r'\bfrais medicaux\b',
            r'\bhospitalisation\b',
        )
        health_matches = [
            pattern for pattern in health_document_patterns
            if re.search(pattern, normalized_name + ' ' + normalized_text)
        ]
        if (
            re.search(r'\b(?:assurance|demande) sante\b|\bmutuelle\b', normalized_name)
            or len(health_matches) >= 2
        ):
            return (
                'unsupported',
                'Cette pièce semble concerner l’assurance santé. Cette démonstration '
                'traite uniquement les dossiers d’assurance automobile ; aucune '
                'recommandation automobile ne doit être tirée de ce document.',
            )

        auto_document_patterns = (
            r'\bassurance automobile\b',
            r'\bassurance auto\b',
            r'\bpolice automobile\b',
            r'\bpolice auto\b',
            r'\bvehicule\b',
            r'\bconducteur\b',
            r'\bimmatriculation\b',
            r'\bsinistre automobile\b',
            r'\baccident de la route\b',
            r'\bautomobile\b',
            r'\bcar policy\b',
            r'\bmotor insurance\b',
            r'\binsured vehicle\b',
        )
        if any(
            re.search(pattern, normalized_name + ' ' + normalized_text)
            for pattern in auto_document_patterns
        ):
            return 'supported', None

        return (
            'unverified',
            'Le type de cette pièce n’a pas pu être confirmé comme un document '
            'd’assurance automobile. Aucune recommandation ne sera proposée avant '
            'la vérification du document par un souscripteur.',
        )

