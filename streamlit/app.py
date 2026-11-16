"""Local Streamlit workbench for the multi-agent underwriting pilot."""

import json
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orchestrator.langgraph_orchestrator import LangGraphOrchestrator
from products.registry import DEFAULT_PRODUCT_LINE, get_product_pack, list_product_lines
from utils.case_store import HUMAN_DECISIONS, CaseStore, role_can
from utils.document_bundle import build_document_bundle, ocr_available


DECISION_LABELS = {
    'Accept_Standard': 'Recommandation favorable',
    'Accept_with_Surcharge': 'Favorable avec surprime',
    'Accept_with_Exclusion': 'Exclusions à vérifier',
    'Reject': 'Refus recommandé — revue obligatoire',
    'Escalate_to_Human': 'Revue humaine requise',
    'Request_More_Info': 'Informations complémentaires',
}
HUMAN_DECISION_LABELS = {
    'Accept_Standard': 'Accepter (standard)',
    'Accept_with_Surcharge': 'Accepter avec surprime',
    'Accept_with_Exclusion': 'Accepter avec exclusion',
    'Reject': 'Refuser',
    'Request_More_Info': 'Demander des pièces / infos',
}
AREA_LABELS = {
    'A': 'A · Rural / très faible densité',
    'B': 'B · Semi-rural',
    'C': 'C · Périurbain',
    'D': 'D · Urbain',
    'E': 'E · Urbain dense',
    'F': 'F · Grande métropole / très dense',
}
VEHICLE_GAS_OPTIONS = ['Regular', 'Diesel']
DOCUMENT_STATUS_LABELS = {
    'text_extracted': 'texte extrait',
    'ocr_extracted': 'texte extrait par OCR',
    'attached_unreadable': 'joint, aucun texte détecté',
    'attached_reader_unavailable': 'joint, lecteur PDF indisponible',
    'attached_ocr_unavailable': 'joint, OCR indisponible',
    'invalid': 'refusé (fichier invalide)',
    'unsupported_format': 'format non pris en charge',
}

st.set_page_config(
    page_title='AURELIA Assurance — Pilotage souscription',
    page_icon='🛡️',
    layout='wide',
)

st.markdown(
    """
    <style>
    :root {
        --ink: #102a36;
        --muted: #647783;
        --paper: #f4f7f8;
        --card: #ffffff;
        --line: #e2e9eb;
        --teal: #087f75;
        --mint: #dff3ed;
        --gold: #d7ae67;
    }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { right: 1.25rem; }
    [data-testid="stSidebar"] {
        background: #102a36;
        border-right: 1px solid rgba(255,255,255,.08);
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }
    [data-testid="stSidebar"] * { color: #eaf3f3; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding: .62rem .72rem;
        border-radius: 12px;
        transition: background .18s ease;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: rgba(255,255,255,.09);
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] input:checked + div {
        color: #80e0c3;
    }
    [data-testid="stMainBlockContainer"] {
        max-width: 1440px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }
    h1, h2, h3, [data-testid="stMetricValue"] {
        color: var(--ink);
        letter-spacing: -.035em;
    }
    h1 { font-size: clamp(2.1rem, 3.3vw, 3.2rem) !important; font-weight: 720 !important; }
    h2 { font-size: 1.55rem !important; font-weight: 680 !important; }
    h3 { font-size: 1.15rem !important; font-weight: 650 !important; }
    p, label, li { color: #435864; }
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        box-shadow: 0 8px 26px rgba(16,42,54,.035);
    }
    [data-testid="stMetricLabel"] { color: var(--muted); font-size: .82rem; }
    [data-testid="stMetricValue"] { font-size: 1.8rem; }
    .brand-mark {
        width: 42px; height: 42px; display: inline-flex; align-items: center;
        justify-content: center; border-radius: 14px; background: #dff3ed;
        color: #087f75; font-size: 23px; margin-right: 10px;
    }
    .brand-row { display: flex; align-items: center; margin: .25rem 0 2rem; }
    .brand-name { color: white; font-size: 1.08rem; line-height: 1.15; font-weight: 700; }
    [data-testid="stMain"] .brand-name { color: var(--ink); }
    .brand-sub { color: #9db4bb; font-size: .72rem; margin-top: 4px; letter-spacing: .05em; }
    .hero {
        position: relative; overflow: hidden; display: flex; align-items: center;
        justify-content: space-between; gap: 2rem; padding: 2.6rem 3rem;
        margin: .15rem 0 1.7rem; border-radius: 28px;
        background: linear-gradient(118deg, #102a36 0%, #154850 65%, #176b67 100%);
        box-shadow: 0 20px 50px rgba(16,42,54,.17);
    }
    .hero:after {
        content: ""; position: absolute; width: 410px; height: 410px; right: 4%;
        top: -225px; border: 1px solid rgba(213,239,231,.12); border-radius: 50%;
        box-shadow: 0 0 0 35px rgba(213,239,231,.035), 0 0 0 75px rgba(213,239,231,.025);
    }
    .hero-copy { position: relative; z-index: 1; max-width: 700px; }
    .hero-kicker { color: #8fe0c6; font-size: .78rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }
    .hero-title { color: white; font-weight: 740; font-size: clamp(2rem, 3.3vw, 3.25rem); letter-spacing: -.045em; line-height: 1.08; margin: .8rem 0; }
    .hero-text { color: #c8d9dc; font-size: 1rem; line-height: 1.6; max-width: 590px; }
    .hero-badge { display: inline-flex; border: 1px solid rgba(143,224,198,.34); color: #b7efdc; background: rgba(12,37,48,.25); border-radius: 999px; padding: .46rem .8rem; font-size: .76rem; margin-top: 1rem; }
    .hero-art { position: relative; z-index: 1; width: min(29%, 245px); min-width: 145px; filter: drop-shadow(0 16px 25px rgba(4,23,29,.22)); }
    .section-label { color: var(--teal); font-size: .73rem; font-weight: 750; letter-spacing: .14em; text-transform: uppercase; margin-bottom: .35rem; }
    .section-intro { color: var(--muted); margin: -.2rem 0 1.25rem; }
    .form-shell {
        background: white; border: 1px solid var(--line); border-radius: 22px;
        padding: 1.5rem 1.6rem; margin: .8rem 0 1rem;
        box-shadow: 0 12px 35px rgba(16,42,54,.045);
    }
    .upload-hint {
        border-left: 3px solid var(--teal); background: #eff8f5;
        color: #43615e; padding: .8rem 1rem; border-radius: 0 12px 12px 0;
        font-size: .88rem; margin: .5rem 0 1rem;
    }
    .result-shell {
        background: white; border: 1px solid var(--line); border-radius: 22px;
        padding: 1.5rem 1.7rem; margin: .9rem 0;
        box-shadow: 0 12px 35px rgba(16,42,54,.04);
    }
    .stButton > button[kind="primaryFormSubmit"],
    [data-testid="stFormSubmitButton"] button {
        background: var(--teal); color: white; border: 0; border-radius: 12px;
        min-height: 3rem; padding: 0 1.4rem; font-weight: 700;
        box-shadow: 0 8px 18px rgba(8,127,117,.18);
    }
    [data-testid="stFormSubmitButton"] button:hover { background: #066d65; color: white; }
    [data-testid="stFileUploader"] section {
        border: 1.5px dashed #9fc5bd; border-radius: 16px; background: #f7fbfa;
    }
    [data-testid="stFileUploader"] section:hover { border-color: var(--teal); }
    [data-testid="stExpander"] {
        border-color: var(--line); border-radius: 14px; background: white;
    }
    [data-testid="stTabs"] button[role="tab"] { font-weight: 650; }
    [data-testid="stAlert"] { border-radius: 14px; }
    code { color: #12675e; }
    @media (max-width: 760px) {
        [data-testid="stMainBlockContainer"] { padding: 1rem .85rem 2rem; }
        .hero { padding: 1.7rem 1.45rem; border-radius: 21px; }
        .hero-art { min-width: 92px; width: 25%; }
        .form-shell, .result-shell { padding: 1.1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_orchestrator():
    return LangGraphOrchestrator()


@st.cache_resource
def get_case_store():
    return CaseStore()


def require_login() -> dict:
    user = st.session_state.get('user')
    if user:
        return user

    st.markdown(
        '<div class="brand-row"><div class="brand-mark">◈</div>'
        '<div><div class="brand-name">AURELIA</div>'
        '<div class="brand-sub">PILOTE SOUSCRIPTION</div></div></div>',
        unsafe_allow_html=True,
    )
    st.title('Connexion souscripteur')
    st.caption(
        'Accès réservé au pilote. Comptes démo : `uw`, `manager`, `compliance`, `admin` '
        '(mot de passe `<rôle>-demo-change-me`, à changer via variables d’environnement). '
        'Rôles : souscripteur (analyse + décision), manager (décision + override), '
        'conformité (lecture seule), admin (gestion des comptes).'
    )
    with st.form('login_form'):
        username = st.text_input('Identifiant')
        password = st.text_input('Mot de passe', type='password')
        submitted = st.form_submit_button('Se connecter', type='primary')
    if submitted:
        authenticated = get_case_store().authenticate(username, password)
        if authenticated is None:
            st.error('Identifiants invalides.')
        else:
            st.session_state['user'] = authenticated
            st.rerun()
    st.stop()


def render_sidebar(user: dict) -> str:
    with st.sidebar:
        st.markdown(
            '<div class="brand-row"><div class="brand-mark">◈</div>'
            '<div><div class="brand-name">AURELIA</div>'
            '<div class="brand-sub">ASSURANCE · PILOTE</div></div></div>',
            unsafe_allow_html=True,
        )
        st.caption(f"{user['username']} · rôle {user['role']}")
        pages = []
        if role_can(user['role'], 'analyze'):
            pages.append('Nouvelle analyse')
        pages.append('File HITL')
        pages.append('Audit / export')
        page = st.radio(
            'Navigation',
            pages,
            label_visibility='collapsed',
        )
        if not role_can(user['role'], 'decide'):
            st.caption('Rôle en lecture seule sur la décision HITL.')
        st.markdown('---')
        st.markdown(
            '<div class="section-label" style="color:#80e0c3">POLITIQUE PILOTE</div>',
            unsafe_allow_html=True,
        )
        st.write(
            'Aide à la décision uniquement. Toute recommandation entre '
            'en file HITL. Le LLM ne décide jamais.'
        )
        if st.button('Se déconnecter', use_container_width=True):
            st.session_state.clear()
            st.rerun()
    return page


def render_hero():
    st.markdown(
        """
        <div class="brand-row">
          <div class="brand-mark">◈</div>
          <div><div class="brand-name">AURELIA <span style="color:#80e0c3">ASSURANCE</span></div>
          <div class="brand-sub">ESPACE DE SOUSCRIPTION · PILOTE HITL</div></div>
        </div>
        <section class="hero">
          <div class="hero-copy">
            <div class="hero-kicker">Aide à la décision · Revue humaine obligatoire</div>
            <div class="hero-title">Votre dossier,<br>analysé avec attention.</div>
            <div class="hero-text">L’IA propose une recommandation. Un souscripteur confirme, ajuste ou refuse avant toute suite opérationnelle. L’auto est le premier vertical ; d’autres lignes pourront être ajoutées plus tard.</div>
            <div class="hero-badge">● &nbsp; Pilote — aucune décision automatisée de production</div>
          </div>
          <svg class="hero-art" viewBox="0 0 240 240" fill="none" xmlns="http://www.w3.org/2000/svg" aria-label="Illustration de protection assurance">
            <circle cx="120" cy="120" r="104" fill="#DFF3ED" fill-opacity=".08" stroke="#DFF3ED" stroke-opacity=".22"/>
            <circle cx="120" cy="120" r="82" fill="#DFF3ED" fill-opacity=".05" stroke="#DFF3ED" stroke-opacity=".25"/>
            <path d="M120 43L174 63V111C174 151 151 178 120 195C89 178 66 151 66 111V63L120 43Z" fill="#E5F3EE" fill-opacity=".96"/>
            <path d="M120 57L161 72V111C161 141 145 163 120 178C95 163 79 141 79 111V72L120 57Z" fill="#0B776F"/>
            <path d="M98 116L113 131L144 96" stroke="#C9F0DF" stroke-width="8" stroke-linecap="round" stroke-linejoin="round"/>
            <circle cx="185" cy="61" r="12" fill="#D9B76D"/><path d="M180 61H190M185 56V66" stroke="#173A43" stroke-width="2" stroke-linecap="round"/>
            <circle cx="52" cy="157" r="7" fill="#83D7BB"/><circle cx="190" cy="166" r="5" fill="#83D7BB"/>
          </svg>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_underwriting_report(report: dict):
    """Render the consolidated underwriting report (bilan de souscription)."""
    st.markdown('<div class="section-label">Bilan de souscription</div>', unsafe_allow_html=True)
    st.subheader(report.get('headline', 'Analyse du dossier'))
    st.caption(
        f"Dossier {report.get('application_id') or '—'} · ligne "
        f"{report.get('product_line') or '—'} · produit le {report.get('generated_at', '—')}"
    )
    for line in report.get('summary_lines', []):
        st.write(f'- {line}')

    risk = report.get('risk_assessment', {})
    uncertainty = report.get('uncertainty', {})
    profile = report.get('profile', {})
    claims = report.get('claims_history', {})
    environment = report.get('environment', {})
    metrics = st.columns(5)
    risk_score = risk.get('final_risk_score')
    metrics[0].metric(
        'Indice de risque',
        f"{float(risk_score):.1f}/100" if isinstance(risk_score, (int, float)) else '—',
    )
    confidence = risk.get('confidence')
    metrics[1].metric(
        'Confiance modèle',
        f"{float(confidence) * 100:.0f}%" if isinstance(confidence, (int, float)) else '—',
    )
    uncertainty_value = uncertainty.get('predictive_uncertainty')
    metrics[2].metric(
        'Incertitude modèle',
        f"{float(uncertainty_value):.2f}" if isinstance(uncertainty_value, (int, float)) else '—',
        help='Entropie normalisée de la distribution prédite (0 = certain, 1 = ambigu).',
    )
    metrics[3].metric('Bande bonus-malus', profile.get('bonus_malus_band') or '—')
    prior_claims = claims.get('prior_claims_count')
    metrics[4].metric(
        'Antécédents au fichier',
        '—' if prior_claims is None else str(prior_claims),
    )

    signals = []
    signals.extend(profile.get('risk_factors') or [])
    signals.extend(claims.get('fraud_indicators') or [])
    signals.extend(environment.get('factors') or [])
    if signals:
        st.write('**Éléments remontés par les agents**')
        for signal in signals:
            st.write(f'- {signal}')

    adjustments = environment.get('adjustments') or []
    if adjustments:
        st.write('**Ajustements suggérés (non tarifaires)**')
        for adjustment in adjustments:
            st.write(f'- {adjustment}')

    documents = report.get('documents', {})
    checklist = documents.get('checklist') or {}
    if checklist.get('required'):
        provided = sum(1 for item in checklist['required'] if item.get('provided'))
        st.write(f"**Pièces du dossier** — {provided}/{len(checklist['required'])} présentes")
        for item in checklist['required']:
            mark = '✅' if item.get('provided') else ('❌' if item.get('mandatory') else '○')
            st.write(f"- {mark} {item['label_fr']}")
    if documents.get('review_terms'):
        st.info('Points à vérifier dans les pièces : ' + ', '.join(documents['review_terms']))

    governance = report.get('governance', {})
    st.write('**Gouvernance**')
    st.write(f"- Statut de conformité : {governance.get('compliance_status') or '—'}")
    st.write(
        "- Revue d’équité : "
        + ('requise' if governance.get('fairness_review_required') else 'non requise')
    )
    st.write(
        "- Décision : humaine obligatoire (HITL) — "
        + ('escalade déclenchée' if governance.get('escalation_triggered') else 'recommandation à confirmer')
    )

    if report.get('next_steps'):
        st.write('**Suite proposée**')
        for step in report['next_steps']:
            st.write(f'- {step}')

    sources = report.get('provenance', {}).get('sources') or []
    if sources:
        st.caption(
            'Sources de référence : '
            + ' · '.join(
                f"{source.get('source_id')} v{source.get('version')}"
                if source.get('available')
                else f"{source.get('source_id')} indisponible"
                for source in sources
            )
        )


def render_decision_summary(result: dict):
    report = result.get('report_result') or {}
    if report:
        render_underwriting_report(report)
        st.divider()
        st.markdown(
            '<div class="section-label">Détail documentaire et traçabilité</div>',
            unsafe_allow_html=True,
        )
    decision = result['decision_result']
    st.markdown('<div class="section-label">Synthèse de l’analyse</div>', unsafe_allow_html=True)
    st.subheader(DECISION_LABELS.get(decision['final_decision'], decision['final_decision']))
    document_result = result['document_result']
    uploaded_doc = document_result.get('uploaded_document', {})
    uploaded_documents = document_result.get('documents') or []
    for document in uploaded_documents:
        status = document.get('extraction_status')
        label = DOCUMENT_STATUS_LABELS.get(status, status)
        line = f"{document['filename']} — {label}"
        if status in {'text_extracted', 'ocr_extracted'}:
            st.success(line)
        else:
            st.warning(line)
    if uploaded_doc.get('provided'):
        if uploaded_doc.get('scope_status') in {'unsupported', 'unverified'}:
            st.warning(uploaded_doc.get('scope_reason'))
        elif not uploaded_documents and uploaded_doc.get('extraction_status') == 'text_extracted':
            st.success(
                f"Texte extrait pour une vérification indicative : "
                f"{uploaded_doc.get('filename')}"
            )
        elif not uploaded_documents:
            st.warning(
                f"Document reçu : {uploaded_doc.get('filename')}. "
                'Son contenu n’a pas pu être lu.'
            )
        possible_terms = uploaded_doc.get('possible_terms_for_human_review', [])
        if possible_terms and uploaded_doc.get('scope_status') == 'supported':
            st.info('Points à vérifier dans le document : ' + ', '.join(possible_terms))
        elif uploaded_doc.get('scope_status') == 'supported':
            st.info(
                'Aucun point de la liste de vérification n’a été repéré. '
                'Cela ne garantit pas que le document ne contient pas d’exclusion.'
            )
    checklist = document_result.get('document_checklist') or {}
    if checklist.get('missing_mandatory'):
        st.error(
            'Pièces obligatoires manquantes : '
            + ', '.join(checklist['missing_mandatory'])
            + '. Le dossier doit être complété avant décision.'
        )
    if decision.get('escalation_triggered'):
        st.warning('Le dossier nécessite un examen par un souscripteur avant toute décision.')
    else:
        st.info('Recommandation de pilote : elle doit être confirmée dans la file HITL.')

    llm_document_result = result.get('llm_document_result', {})
    if llm_document_result.get('status') == 'analyzed':
        st.markdown('### Lecture documentaire assistée')
        st.caption(
            'Synthèse générée à partir des extraits retrouvés. Vérifie chaque '
            'citation dans le document original ; cette analyse ne détermine pas '
            'la décision de souscription.'
        )
        st.write(llm_document_result['summary'])
        for finding in llm_document_result['findings']:
            st.markdown(
                f"**{finding['category'].capitalize()} — {finding['finding']}**"
            )
            st.markdown(f"> {finding['evidence_quote']}")
        if llm_document_result['uncertainties']:
            st.markdown('**À vérifier**')
            for uncertainty in llm_document_result['uncertainties']:
                st.write(f'- {uncertainty}')
    elif llm_document_result.get('status') == 'no_relevant_passages':
        st.info(
            'La revue LLM n’a trouvé aucun passage pertinent à examiner. '
            'Aucune conclusion documentaire n’est disponible.'
        )

    explanation = result['explanation_result']['explanations']['customer']
    st.markdown(f"**{explanation['summary']}**")
    for detail in explanation['details']:
        st.write(f'- {detail}')
    st.write('**Suite proposée**')
    for step in explanation['next_steps']:
        st.write(f'- {step}')


def render_application_page(user: dict):
    render_hero()
    product_pack = get_product_pack(DEFAULT_PRODUCT_LINE)
    st.markdown(
        '<div class="section-label">Espace de souscription</div>'
        '<h1 style="margin-top:0">Créer une analyse de dossier</h1>'
        '<p class="section-intro">Complète les informations demandées et joins un document. '
        'La recommandation est enregistrée et envoyée en file HITL.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="upload-hint"><strong>Protection des données</strong> · '
        'N’utilise pas de données personnelles réelles. Les dossiers du pilote sont '
        'conservés localement dans SQLite pour la revue humaine et l’audit. '
        'Le LLM Groq reste désactivé par défaut ; s’il est activé, des extraits quittent '
        'l’application.</div>',
        unsafe_allow_html=True,
    )

    enabled_lines = list_product_lines(enabled_only=True)
    line_labels = {item['code']: item['label_fr'] for item in enabled_lines}
    reserved = [
        item for item in list_product_lines(enabled_only=False) if item['status'] != 'active'
    ]

    st.markdown(
        '<div class="section-label">01 · Informations du dossier</div>'
        '<h3>Informations du dossier</h3>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Produit actif : {product_pack['label_fr']} · règles {product_pack['version']}. "
        'Habitation et autres lignes : extension ultérieure.'
    )
    product_line = st.selectbox(
        'Ligne de produit',
        options=list(line_labels),
        format_func=lambda code: line_labels[code],
        index=0,
    )
    if reserved:
        st.caption(
            'Réservé plus tard : '
            + ', '.join(f"{item['label_fr']} ({item['code']})" for item in reserved)
        )
    left, right = st.columns(2)
    with left:
        application_id = st.text_input('Référence du dossier', value='DEMO-001', max_chars=64)
        driver_age = st.number_input('Âge du conducteur', min_value=18, max_value=100, value=35)
        bonus_malus = st.number_input(
            'Coefficient bonus-malus (CRM)',
            min_value=50.0,
            max_value=350.0,
            value=60.0,
            step=5.0,
            help='Coefficient réel de réduction/majoration : plus il est faible, meilleur est le conducteur.',
        )
        vehicle_age = st.number_input('Âge du véhicule (années)', min_value=0, max_value=40, value=5)
    with right:
        vehicle_power = st.number_input('Puissance fiscale (CV)', min_value=4, max_value=20, value=6)
        vehicle_gas = st.selectbox('Carburant', VEHICLE_GAS_OPTIONS)
        density = st.number_input(
            'Densité de population de la commune',
            min_value=0.0,
            max_value=100000.0,
            value=1200.0,
            step=100.0,
        )
        area_label = st.selectbox(
            'Classe géographique (freMTPL2)',
            list(AREA_LABELS),
            format_func=lambda code: AREA_LABELS[code],
            help='Classe réelle de densité de la commune, de A (rural) à F (métropole dense).',
        )

    st.markdown(
        '<div class="section-label">02 · Pièces justificatives</div>'
        '<h3>Pièces du dossier</h3>',
        unsafe_allow_html=True,
    )
    required_documents = product_pack['required_documents']
    type_choices = [doc['label_fr'] for doc in required_documents] + ['Autre / non précisé']
    label_to_code = {doc['label_fr']: doc['code'] for doc in required_documents}
    st.caption(
        'Checklist produit : '
        + ', '.join(
            f"{doc['label_fr']}{' *' if doc.get('mandatory') else ''}"
            for doc in required_documents
        )
        + ' · * pièce obligatoire'
    )
    uploaded_documents = st.file_uploader(
        'Déposer une ou plusieurs pièces',
        type=['pdf', 'txt', 'png', 'jpg', 'jpeg', 'tiff', 'bmp', 'webp'],
        accept_multiple_files=True,
        help=(
            'PDF texte ou TXT UTF-8 · images (png/jpg/tiff) lues par OCR si les '
            'dépendances optionnelles sont installées · 10 Mo maximum par pièce.'
        ),
    )
    if not ocr_available():
        st.caption(
            'OCR indisponible dans cet environnement : les PDF scannés et les images '
            'seront joints mais non lus. Installe `requirements-research.txt` pour activer l’OCR.'
        )

    document_payload = []
    bundle_preview = None
    if uploaded_documents:
        for index, uploaded in enumerate(uploaded_documents):
            info_column, type_column = st.columns([3, 2])
            size_bytes = uploaded.size or 0
            size_label = (
                f"{size_bytes} o" if size_bytes < 1024
                else f"{size_bytes / 1024:.0f} Ko"
            )
            with info_column:
                st.write(
                    f"**{Path(uploaded.name).name}** · {size_label}"
                )
            with type_column:
                declared_label = st.selectbox(
                    'Type de pièce',
                    type_choices,
                    index=len(type_choices) - 1,
                    key=f'declared_type_{index}',
                    label_visibility='collapsed',
                )
            document_payload.append(
                {
                    'filename': Path(uploaded.name).name,
                    'content': uploaded.getvalue(),
                    'declared_type': label_to_code.get(declared_label),
                }
            )
        bundle_preview = build_document_bundle(document_payload, product_line=product_line)
        for document in bundle_preview['documents']:
            status_label = DOCUMENT_STATUS_LABELS.get(
                document['extraction_status'], document['extraction_status']
            )
            st.caption(f"· {document['filename']} — {status_label}")
        for item in bundle_preview['checklist']['required']:
            mark = '✅' if item['provided'] else ('❌' if item['mandatory'] else '○')
            st.caption(f"{mark} {item['label_fr']}")

    use_llm_document_analysis = st.checkbox(
        'Activer la revue documentaire par LLM (Groq)',
        value=False,
        help='Option facultative. Le modèle ne prend pas de décision de souscription.',
    )
    st.caption(
        'En activant cette option, seuls des extraits de texte retrouvés dans les '
        'pièces sont transmis à Groq. Utilise uniquement des documents fictifs.'
    )
    collaboration_mode = st.selectbox(
        'Mode de collaboration des agents',
        options=('sequential', 'parallel'),
        index=0,
        help=(
            'Séquentiel : les agents s’exécutent dans un ordre strict. '
            'Parallèle : profil, documents, environnement et antécédents sont '
            'traités simultanément avant le risque et la décision.'
        ),
    )
    submitted = st.button(
        'Analyser et envoyer en HITL  →',
        type='primary',
        use_container_width=True,
    )

    if submitted:
        if not uploaded_documents:
            st.warning('Ajoute au moins une pièce (PDF, TXT ou image) pour lancer l’analyse.')
            return
        missing = bundle_preview['checklist']['missing_mandatory']
        if missing:
            missing_labels = [
                item['label_fr']
                for item in bundle_preview['checklist']['required']
                if item['code'] in missing
            ]
            st.error(
                'Pièces obligatoires manquantes : ' + ', '.join(missing_labels) + '. '
                'Renseigne le type de chaque pièce ou ajoute les pièces requises.'
            )
            return

        application = {
            'application_id': application_id.strip() or 'DEMO-001',
            'product_line': product_line,
            'driver_age': int(driver_age),
            'bonus_malus': float(bonus_malus),
            'vehicle_age': int(vehicle_age),
            'vehicle_power': int(vehicle_power),
            'vehicle_gas': vehicle_gas,
            'density': float(density),
            'area': area_label,
        }
        try:
            with st.spinner('Analyse du dossier en cours…'):
                result = get_orchestrator().run_workflow(
                    application,
                    mode=collaboration_mode,
                    decision_mode='human_in_the_loop',
                    document_bundle=bundle_preview,
                    enable_llm_document_analysis=use_llm_document_analysis,
                )
                case = get_case_store().create_case(
                    application=application,
                    result=result,
                    product_line=product_line,
                    actor=user['username'],
                    llm_enabled=use_llm_document_analysis,
                )
        except (FileNotFoundError, ImportError, ValueError, RuntimeError) as error:
            st.error(f'Analyse impossible : {error}')
            return

        st.session_state['latest_case_id'] = case['id']
        st.session_state['latest_result'] = {'result': result, 'case': case}
        st.success(
            f"Dossier enregistré · case `{case['id']}` envoyé en file HITL "
            f"(réf. {case['application_id']})."
        )

    latest = st.session_state.get('latest_result')
    if not latest:
        st.info('Complète les informations et joins une pièce pour afficher la synthèse du dossier.')
        return

    st.divider()
    render_decision_summary(latest['result'])
    case = latest.get('case')
    if case:
        audit = get_case_store().export_audit(case['id'])
        st.download_button(
            'Télécharger la trace d’audit (JSON)',
            data=json.dumps(audit, ensure_ascii=False, indent=2),
            file_name=f"audit_{case['application_id']}_{case['id'][:8]}.json",
            mime='application/json',
            use_container_width=True,
        )


def render_hitl_page(user: dict):
    st.markdown(
        '<div class="section-label">Revue humaine</div>'
        '<h1 style="margin-top:0">File HITL</h1>'
        '<p class="section-intro">Confirme, ajuste ou refuse la recommandation IA. '
        'La décision humaine devient la source de vérité du dossier.</p>',
        unsafe_allow_html=True,
    )
    store = get_case_store()
    pending = store.list_pending_cases()
    left, right, third = st.columns(3)
    left.metric('En attente', len(pending))
    right.metric('Décidés (aperçu)', len([c for c in store.list_recent_cases(100) if c['status'] == 'decided']))
    third.metric('Produit actif', 'Automobile')

    if not pending:
        st.info('Aucun dossier en attente. Lance une analyse pour alimenter la file.')
        return

    labels = {
        case['id']: (
            f"{case['application_id']} · {case['ai_decision']} · "
            f"{case['created_at']} · {case['id'][:8]}"
        )
        for case in pending
    }
    selected_id = st.selectbox(
        'Dossier à traiter',
        options=list(labels),
        format_func=lambda case_id: labels[case_id],
    )
    case = store.get_case(selected_id)
    if case is None:
        st.error('Dossier introuvable.')
        return

    st.markdown(f"**Référence** `{case['application_id']}` · produit `{case['product_line']}`")
    st.write(
        f"Recommandation IA : **{DECISION_LABELS.get(case['ai_decision'], case['ai_decision'])}**"
    )
    with st.expander('Détail de la recommandation IA', expanded=True):
        render_decision_summary(case['result'])

    if not role_can(user['role'], 'decide'):
        st.info(
            'Ton rôle est en lecture seule sur la file HITL : la décision humaine est '
            'réservée aux souscripteurs, managers et admins. Le dossier et la trace '
            'd’audit restent consultables.'
        )
        return

    with st.form('human_decision_form'):
        human_decision = st.selectbox(
            'Décision humaine',
            options=sorted(HUMAN_DECISIONS),
            format_func=lambda code: HUMAN_DECISION_LABELS.get(code, code),
        )
        human_reason = st.text_area(
            'Motif (obligatoire)',
            placeholder='Ex. confirmation après revue du relevé d’information et de la pièce jointe.',
            height=120,
        )
        submitted = st.form_submit_button('Enregistrer la décision humaine', type='primary')
    if submitted:
        try:
            updated = store.record_human_decision(
                case_id=case['id'],
                human_decision=human_decision,
                human_reason=human_reason,
                actor=user['username'],
            )
        except ValueError as error:
            st.error(str(error))
            return
        st.success(
            f"Décision humaine enregistrée : {HUMAN_DECISION_LABELS.get(updated['human_decision'], updated['human_decision'])}"
        )
        st.rerun()


def render_audit_page(user: dict):
    st.markdown(
        '<div class="section-label">Gouvernance</div>'
        '<h1 style="margin-top:0">Audit / export</h1>'
        '<p class="section-intro">Exporte la trace d’un dossier : recommandation IA, '
        'décision humaine, événements append-only.</p>',
        unsafe_allow_html=True,
    )
    store = get_case_store()
    cases = store.list_recent_cases(limit=100)
    if not cases:
        st.info('Aucun dossier enregistré pour le moment.')
        return

    labels = {
        case['id']: (
            f"{case['application_id']} · {case['status']} · "
            f"IA={case['ai_decision']} · humain={case['human_decision'] or '—'} · "
            f"{case['created_at']}"
        )
        for case in cases
    }
    selected_id = st.selectbox(
        'Dossier',
        options=list(labels),
        format_func=lambda case_id: labels[case_id],
    )
    case = store.get_case(selected_id)
    if case is None:
        st.error('Dossier introuvable.')
        return

    col1, col2, col3 = st.columns(3)
    col1.write(f"**Statut** : {case['status']}")
    col2.write(f"**IA** : {case['ai_decision']}")
    col3.write(f"**Humain** : {case['human_decision'] or '—'}")
    if case['human_reason']:
        st.write(f"**Motif humain** : {case['human_reason']}")
        st.caption(f"Décidé par {case['decided_by']} le {case['decided_at']}")

    audit = store.export_audit(case['id'])
    chain = audit.get('audit_chain', {})
    if chain.get('valid'):
        st.success(
            f"Chaîne d’audit vérifiée (append-only) · {chain.get('events', 0)} événement(s)."
        )
    else:
        st.error(
            f"Chaîne d’audit invalide : {chain.get('reason')} "
            f"(index {chain.get('at_index')})."
        )
    st.download_button(
        'Télécharger l’audit JSON',
        data=json.dumps(audit, ensure_ascii=False, indent=2),
        file_name=f"audit_{case['application_id']}_{case['id'][:8]}.json",
        mime='application/json',
        use_container_width=True,
    )
    st.json(
        {
            'case_id': audit['case_id'],
            'exported_at': audit['exported_at'],
            'events': [
                {
                    'event_type': event['event_type'],
                    'actor': event['actor'],
                    'created_at': event['created_at'],
                }
                for event in audit['audit_events']
            ],
        }
    )
    if user['role'] != 'admin':
        st.caption('Export disponible pour tous les rôles pilote ; droits admin élargis plus tard.')


def main():
    user = require_login()
    page = render_sidebar(user)
    if page == 'Nouvelle analyse':
        render_application_page(user)
    elif page == 'File HITL':
        render_hitl_page(user)
    else:
        render_audit_page(user)


main()
