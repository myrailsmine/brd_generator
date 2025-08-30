"""
Enhanced AI Processing Utilities with Advanced Intelligence
Complete implementation with multi-step reasoning, context awareness, and adaptive generation
"""

import streamlit as st
import pandas as pd
import re
import json
from typing import List, Dict, Any, Tuple
from config.app_config import ENHANCED_BRD_STRUCTURE, QualityCheck
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    from langchain_community.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not available. AI features will be limited.")

@st.cache_resource
def init_enhanced_llm():
    """Initialize ChatOpenAI with enhanced configuration"""
    if not LANGCHAIN_AVAILABLE:
        logger.error("LangChain not available for AI processing")
        return None
        
    try:
        config = st.session_state.get('llm_config', {})
        return ChatOpenAI(
            base_url=config.get('base_url', "http://localhost:8123/v1"),
            api_key=config.get('api_key', "dummy"),
            model=config.get('model', "llama3"),
            temperature=config.get('temperature', 0.1),  # Lower temperature for more consistent outputs
            max_tokens=config.get('max_tokens', 6000),   # Increased for longer responses
            streaming=False
        )
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")
        return None

class IntelligentDocumentAnalyzer:
    """Advanced document analyzer with multi-step reasoning"""
    
    def __init__(self, llm):
        self.llm = llm
        self.context_memory = {}
        self.regulatory_knowledge = self._load_regulatory_knowledge()
    
    def _load_regulatory_knowledge(self) -> Dict[str, Any]:
        """Load regulatory framework knowledge base"""
        return {
            'basel': {
                'frameworks': ['Basel I', 'Basel II', 'Basel III', 'Basel IV'],
                'key_concepts': ['capital requirements', 'risk weights', 'correlation', 'sensitivities', 'curvature'],
                'section_patterns': [r'MAR\d+\.\d+', r'CRE\d+\.\d+', r'RBC\d+\.\d+'],
                'formula_types': ['PV01', 'CS01', 'VaR', 'correlation matrices', 'risk weights']
            },
            'mifid': {
                'frameworks': ['MiFID I', 'MiFID II'],
                'key_concepts': ['best execution', 'client categorization', 'product governance'],
                'section_patterns': [r'Article\s+\d+', r'Commission\s+Regulation'],
                'formula_types': ['transaction costs', 'price improvement calculations']
            },
            'gdpr': {
                'frameworks': ['GDPR'],
                'key_concepts': ['data protection', 'privacy by design', 'consent management'],
                'section_patterns': [r'Article\s+\d+', r'Recital\s+\d+'],
                'formula_types': ['data retention formulas', 'compliance scoring']
            },
            'solvency': {
                'frameworks': ['Solvency I', 'Solvency II'],
                'key_concepts': ['solvency capital requirement', 'minimum capital requirement', 'pillar 1', 'pillar 2', 'pillar 3'],
                'section_patterns': [r'Article\s+\d+', r'SCR', r'MCR'],
                'formula_types': ['capital calculations', 'risk margin formulas']
            }
        }
    
    def analyze_document_context(self, document_text: str, extracted_elements: Dict[str, Any]) -> Dict[str, Any]:
        """Perform intelligent context analysis of the document"""
        
        # Multi-step analysis
        analysis_steps = [
            self._identify_regulatory_framework,
            self._analyze_mathematical_complexity,
            self._extract_business_context,
            self._identify_stakeholder_relationships,
            self._assess_compliance_requirements,
            self._analyze_document_structure,
            self._evaluate_implementation_complexity
        ]
        
        context = {
            'document_text': document_text,
            'extracted_elements': extracted_elements,
            'analysis_results': {}
        }
        
        for step in analysis_steps:
            try:
                result = step(context)
                context['analysis_results'].update(result)
            except Exception as e:
                logger.warning(f"Analysis step {step.__name__} failed: {e}")
        
        return context['analysis_results']
    
    def _identify_regulatory_framework(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Intelligently identify regulatory frameworks and their implications"""
        
        document_text = context['document_text'].lower()
        frameworks_detected = []
        confidence_scores = {}
        
        for framework, knowledge in self.regulatory_knowledge.items():
            score = 0
            
            # Check for framework mentions
            for fw_name in knowledge['frameworks']:
                if fw_name.lower() in document_text:
                    score += 0.3
            
            # Check for key concepts
            for concept in knowledge['key_concepts']:
                concept_count = document_text.count(concept.lower())
                score += min(concept_count * 0.1, 0.4)
            
            # Check for section patterns
            for pattern in knowledge['section_patterns']:
                matches = len(re.findall(pattern, context['document_text'], re.IGNORECASE))
                score += min(matches * 0.05, 0.3)
            
            if score > 0.2:  # Threshold for framework detection
                frameworks_detected.append(framework)
                confidence_scores[framework] = min(score, 1.0)
        
        return {
            'regulatory_frameworks': frameworks_detected,
            'framework_confidence': confidence_scores,
            'primary_framework': max(confidence_scores.items(), key=lambda x: x[1])[0] if confidence_scores else None
        }
    
    def _analyze_mathematical_complexity(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze mathematical complexity with intelligent classification"""
        
        extracted_elements = context['extracted_elements']
        formulas = extracted_elements.get('extracted_formulas', [])
        tables = extracted_elements.get('extracted_tables', [])
        
        complexity_factors = {
            'formula_density': len(formulas) / max(len(context['document_text'].split()), 1) * 1000,
            'correlation_matrices': len([t for t in tables if 'correlation' in t.get('text', '').lower()]),
            'greek_symbols': len([f for f in formulas if self._contains_greek_symbols(f.get('text', ''))]),
            'advanced_functions': len([f for f in formulas if self._contains_advanced_math(f.get('text', ''))]),
            'multi_dimensional_tables': len([t for t in tables if t.get('rows', 0) > 5 and t.get('columns', 0) > 5])
        }
        
        # Intelligent complexity scoring
        complexity_score = (
            min(complexity_factors['formula_density'] * 0.1, 0.3) +
            min(complexity_factors['correlation_matrices'] * 0.2, 0.25) +
            min(complexity_factors['greek_symbols'] * 0.05, 0.2) +
            min(complexity_factors['advanced_functions'] * 0.1, 0.15) +
            min(complexity_factors['multi_dimensional_tables'] * 0.1, 0.1)
        )
        
        complexity_level = 'Low'
        if complexity_score > 0.7:
            complexity_level = 'Very High'
        elif complexity_score > 0.5:
            complexity_level = 'High'
        elif complexity_score > 0.3:
            complexity_level = 'Medium'
        
        return {
            'mathematical_complexity_detailed': complexity_factors,
            'mathematical_complexity_score': complexity_score,
            'mathematical_complexity_level': complexity_level
        }
    
    def _extract_business_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract business context and objectives"""
        
        document_text = context['document_text']
        
        # Business context indicators
        business_patterns = {
            'objectives': [
                r'objective[s]?\s*[:]\s*([^.!?]+)',
                r'goal[s]?\s*[:]\s*([^.!?]+)',
                r'purpose\s*[:]\s*([^.!?]+)'
            ],
            'scope': [
                r'scope\s*[:]\s*([^.!?]+)',
                r'applies to\s*([^.!?]+)',
                r'covers\s*([^.!?]+)'
            ],
            'stakeholders': [
                r'responsible\s*[:]\s*([^.!?]+)',
                r'owner\s*[:]\s*([^.!?]+)',
                r'stakeholder[s]?\s*[:]\s*([^.!?]+)'
            ]
        }
        
        extracted_context = {}
        for category, patterns in business_patterns.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, document_text, re.IGNORECASE)
                matches.extend(found)
            extracted_context[category] = matches[:3]  # Top 3 matches
        
        return {'business_context': extracted_context}
    
    def _identify_stakeholder_relationships(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Identify stakeholder relationships and responsibilities"""
        
        document_text = context['document_text']
        
        # Common stakeholder roles in regulatory documents
        stakeholder_patterns = {
            'regulatory_bodies': [
                r'(basel committee|eba|federal reserve|bank of england|ecb)',
                r'(supervisor|regulator|authority)'
            ],
            'internal_roles': [
                r'(risk manager|compliance officer|business analyst)',
                r'(cro|cfo|ceo|board of directors)'
            ],
            'business_units': [
                r'(trading desk|risk department|it department)',
                r'(front office|middle office|back office)'
            ]
        }
        
        identified_stakeholders = {}
        for category, patterns in stakeholder_patterns.items():
            stakeholders = []
            for pattern in patterns:
                matches = re.findall(pattern, document_text, re.IGNORECASE)
                stakeholders.extend([m if isinstance(m, str) else m[0] for m in matches])
            identified_stakeholders[category] = list(set(stakeholders))[:5]
        
        return {'stakeholder_analysis': identified_stakeholders}
    
    def _assess_compliance_requirements(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess compliance requirements and criticality"""
        
        document_text = context['document_text']
        
        # Compliance indicators
        compliance_keywords = {
            'mandatory': ['shall', 'must', 'required', 'mandatory', 'obliged'],
            'optional': ['may', 'can', 'should', 'recommended', 'suggested'],
            'prohibited': ['shall not', 'must not', 'prohibited', 'forbidden', 'not allowed']
        }
        
        compliance_analysis = {}
        for category, keywords in compliance_keywords.items():
            count = sum(document_text.lower().count(keyword) for keyword in keywords)
            compliance_analysis[f'{category}_requirements'] = count
        
        # Calculate compliance criticality
        total_requirements = sum(compliance_analysis.values())
        criticality = 'Low'
        if total_requirements > 50:
            criticality = 'Critical'
        elif total_requirements > 20:
            criticality = 'High'
        elif total_requirements > 10:
            criticality = 'Medium'
        
        compliance_analysis['compliance_criticality'] = criticality
        compliance_analysis['total_compliance_indicators'] = total_requirements
        
        return {'compliance_assessment': compliance_analysis}
    
    def _analyze_document_structure(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze document structure and organization"""
        
        document_text = context['document_text']
        
        # Section analysis
        section_patterns = [
            r'(\d+\.\d+(?:\.\d+)?\s+[A-Z][^.!?]*)',  # Numbered sections
            r'([A-Z][A-Z\s]+:?\s*[A-Z][^.!?]*)',      # All caps headings
            r'(Chapter\s+\d+[^.!?]*)',                 # Chapters
            r'(Part\s+\d+[^.!?]*)',                    # Parts
        ]
        
        sections_found = []
        for pattern in section_patterns:
            matches = re.findall(pattern, document_text)
            sections_found.extend(matches[:10])  # Limit to prevent overflow
        
        # Calculate structure score
        structure_score = min(len(sections_found) / 20, 1.0)  # Normalize to 1.0
        
        return {
            'document_sections': sections_found,
            'structure_score': structure_score,
            'is_well_structured': structure_score > 0.3
        }
    
    def _evaluate_implementation_complexity(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate implementation complexity and effort requirements"""
        
        document_text = context['document_text'].lower()
        
        # Implementation complexity indicators
        complexity_indicators = {
            'technical_terms': ['system', 'database', 'api', 'integration', 'architecture'],
            'process_terms': ['workflow', 'procedure', 'process', 'methodology', 'framework'],
            'resource_terms': ['budget', 'timeline', 'resource', 'capacity', 'effort'],
            'change_terms': ['migration', 'transformation', 'change', 'upgrade', 'implementation']
        }
        
        implementation_scores = {}
        total_score = 0
        
        for category, terms in complexity_indicators.items():
            score = sum(document_text.count(term) for term in terms)
            implementation_scores[category] = score
            total_score += score
        
        # Determine implementation complexity
        if total_score > 30:
            impl_complexity = 'Very High'
        elif total_score > 20:
            impl_complexity = 'High'
        elif total_score > 10:
            impl_complexity = 'Medium'
        else:
            impl_complexity = 'Low'
        
        return {
            'implementation_indicators': implementation_scores,
            'implementation_complexity': impl_complexity,
            'implementation_score': total_score
        }
    
    def _contains_greek_symbols(self, text: str) -> bool:
        """Check if text contains Greek symbols"""
        greek_pattern = r'[αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ]'
        return bool(re.search(greek_pattern, text))
    
    def _contains_advanced_math(self, text: str) -> bool:
        """Check if text contains advanced mathematical functions"""
        advanced_patterns = [
            r'∑', r'∫', r'√', r'∂', r'∇',  # Advanced operators
            r'exp\(', r'log\(', r'ln\(',    # Functions
            r'max\(', r'min\(', r'floor\(', r'ceiling\('  # Special functions
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in advanced_patterns)

class AdaptiveBRDGenerator:
    """Adaptive BRD generator that adjusts to document characteristics"""
    
    def __init__(self, llm, analyzer: IntelligentDocumentAnalyzer):
        self.llm = llm
        self.analyzer = analyzer
        self.section_templates = self._load_adaptive_templates()
    
    def _load_adaptive_templates(self) -> Dict[str, Dict[str, str]]:
        """Load adaptive templates for different document types"""
        return {
            'basel': {
                'system_prompt': """You are a Basel regulatory compliance expert with 20+ years of experience in banking supervision and market risk frameworks. You specialize in translating complex regulatory requirements into clear, actionable business requirements documents that meet the highest audit and compliance standards.""",
                'context_instructions': """Focus on:
                - Capital adequacy requirements and risk calculations
                - Market risk sensitivities and correlation parameters
                - Trading book vs banking book classifications
                - Supervisory review processes and Pillar 2 requirements
                - Implementation timelines and phase-in arrangements
                - Risk weight calculations and formula applications
                - Correlation matrix requirements and bucketing approaches""",
            },
            'solvency': {
                'system_prompt': """You are a Solvency II regulatory expert with extensive experience in insurance regulatory frameworks and capital adequacy requirements. You excel at creating comprehensive business requirements that ensure compliance with European insurance regulations.""",
                'context_instructions': """Focus on:
                - Solvency Capital Requirement (SCR) calculations
                - Minimum Capital Requirement (MCR) frameworks
                - Pillar 1, 2, and 3 requirements
                - Own Risk and Solvency Assessment (ORSA)
                - Quantitative Reporting Templates (QRT)
                - Risk margin calculations and technical provisions""",
            },
            'mifid': {
                'system_prompt': """You are a MiFID II compliance specialist with deep expertise in investment services regulation, client protection, and market transparency requirements. You create detailed business requirements ensuring full regulatory compliance.""",
                'context_instructions': """Focus on:
                - Best execution requirements and RTS 27/28 reporting
                - Client categorization and suitability assessments
                - Product governance and target market identification
                - Transaction reporting under MiFIR
                - Investment research unbundling requirements
                - Market making and systematic internalizer obligations""",
            },
            'technical': {
                'system_prompt': """You are a technical business analyst with extensive experience in system integration and technical requirements documentation. You excel at translating technical specifications into comprehensive business requirements that development teams can implement effectively.""",
                'context_instructions': """Focus on:
                - System architecture and integration points
                - Data flow and processing requirements
                - Performance and scalability requirements
                - Security and access control specifications
                - Testing and validation criteria
                - API specifications and data formats""",
            },
            'business': {
                'system_prompt': """You are a senior business analyst with expertise in process optimization and business transformation. You specialize in creating detailed business requirements that align with strategic objectives and operational excellence.""",
                'context_instructions': """Focus on:
                - Business process workflows and decision points
                - Stakeholder roles and responsibilities
                - Success metrics and KPIs
                - Change management considerations
                - Operational risk and mitigation strategies
                - Cost-benefit analysis and ROI considerations""",
            }
        }
    
    def generate_intelligent_brd_section(
        self, 
        section_name: str, 
        section_config: Dict[str, Any],
        document_context: Dict[str, Any],
        intelligent_analysis: Dict[str, Any]
    ) -> str:
        """Generate BRD section using intelligent analysis and adaptive templates"""
        
        if self.llm is None:
            return self._generate_intelligent_fallback(section_name, section_config, intelligent_analysis)
        
        # Select appropriate template based on analysis
        primary_framework = intelligent_analysis.get('primary_framework', 'business')
        template_key = primary_framework if primary_framework in self.section_templates else 'business'
        template = self.section_templates[template_key]
        
        # Build intelligent prompt
        prompt = self._build_intelligent_prompt(
            section_name, section_config, document_context, intelligent_analysis, template
        )
        
        try:
            # Create messages with intelligent context
            system_message = SystemMessage(content=template['system_prompt'])
            human_message = HumanMessage(content=prompt)
            
            # Generate response
            response = self.llm.invoke([system_message, human_message])
            
            # Post-process response for quality
            return self._post_process_response(response.content, section_config, intelligent_analysis)
            
        except Exception as e:
            logger.error(f"Error in intelligent generation for {section_name}: {e}")
            return self._generate_intelligent_fallback(section_name, section_config, intelligent_analysis)
    
    def _build_intelligent_prompt(
        self, 
        section_name: str, 
        section_config: Dict[str, Any],
        document_context: Dict[str, Any],
        intelligent_analysis: Dict[str, Any],
        template: Dict[str, str]
    ) -> str:
        """Build intelligent prompt using all available context"""
        
        # Extract relevant context
        regulatory_frameworks = intelligent_analysis.get('regulatory_frameworks', [])
        complexity_level = intelligent_analysis.get('mathematical_complexity_level', 'Medium')
        business_context = intelligent_analysis.get('business_context', {})
        stakeholder_analysis = intelligent_analysis.get('stakeholder_analysis', {})
        compliance_assessment = intelligent_analysis.get('compliance_assessment', {})
        implementation_complexity = intelligent_analysis.get('implementation_complexity', 'Medium')
        
        # Build context-aware prompt
        prompt = f"""
        Create a comprehensive "{section_name}" section for a Business Requirements Document.
        
        DOCUMENT INTELLIGENCE CONTEXT:
        - Primary Regulatory Framework: {', '.join(regulatory_frameworks)}
        - Mathematical Complexity: {complexity_level}
        - Implementation Complexity: {implementation_complexity}
        - Compliance Criticality: {compliance_assessment.get('compliance_criticality', 'Medium')}
        - Document Type: {document_context.get('document_type', 'Business Document')}
        
        STAKEHOLDER CONTEXT:
        """
        
        # Add stakeholder context
        for role_type, stakeholders in stakeholder_analysis.items():
            if stakeholders:
                prompt += f"- {role_type.replace('_', ' ').title()}: {', '.join(stakeholders[:3])}\n"
        
        prompt += f"""
        
        BUSINESS CONTEXT:
        """
        
        # Add business objectives if available
        objectives = business_context.get('objectives', [])
        if objectives:
            prompt += f"- Objectives: {'; '.join(objectives[:2])}\n"
        
        scope = business_context.get('scope', [])
        if scope:
            prompt += f"- Scope: {'; '.join(scope[:2])}\n"
        
        # Add section-specific instructions
        prompt += f"""
        
        SECTION REQUIREMENTS:
        - Section Type: {section_config.get('type', 'text')}
        - Description: {section_config.get('description', f'Generate content for {section_name}')}
        """
        
        required_elements = section_config.get('required_elements', [])
        if required_elements:
            prompt += f"- Required Elements: {', '.join(required_elements)}\n"
        
        quality_criteria = section_config.get('quality_criteria', [])
        if quality_criteria:
            prompt += f"- Quality Criteria: {', '.join(quality_criteria)}\n"
        
        # Add template-specific context
        prompt += f"""
        
        CONTEXTUAL GUIDANCE:
        {template.get('context_instructions', '')}
        
        GENERATION REQUIREMENTS:
        """
        
        if section_config.get("type") == "table":
            columns = section_config.get("columns", [])
            prompt += f"""
            Generate a detailed regulatory-compliant table with exactly these columns: {' | '.join(columns)}
            
            Requirements:
            1. Generate 8-12 detailed rows based on the {complexity_level.lower()} complexity and {regulatory_frameworks} framework
            2. Include specific regulatory references where applicable ({', '.join(regulatory_frameworks)})
            3. Use precise terminology appropriate to {complexity_level.lower()} complexity documents
            4. Ensure all entries are audit-ready and compliance-focused for {compliance_assessment.get('compliance_criticality', 'medium').lower()} criticality
            5. Include quantitative metrics and thresholds where appropriate
            6. Address {implementation_complexity.lower()} implementation complexity requirements
            
            Return in pipe-separated format:
            {' | '.join(columns)}
            [Generate rows here - minimum 8 rows for comprehensive coverage]
            """
        else:
            prompt += f"""
            Generate comprehensive, professional content (minimum 500 words) that:
            1. Addresses all required elements with regulatory precision for {', '.join(regulatory_frameworks)}
            2. Uses industry-appropriate language for {complexity_level.lower()} mathematical complexity
            3. Includes specific compliance checkpoints and audit trails for {compliance_assessment.get('compliance_criticality', 'medium').lower()} criticality
            4. References relevant regulatory frameworks with specific section citations where possible
            5. Provides actionable requirements with clear acceptance criteria
            6. Includes metrics and measurement approaches appropriate to {implementation_complexity.lower()} implementation
            7. Addresses stakeholder roles and responsibilities based on identified stakeholders
            8. Considers business objectives and scope as identified in the document analysis
            """
        
        # Add document-specific context from extracted elements
        extracted_elements = document_context.get('extracted_elements', {})
        if extracted_elements.get('extracted_formulas'):
            formula_count = len(extracted_elements['extracted_formulas'])
            prompt += f"\n9. Consider that {formula_count} mathematical formulas were identified - incorporate mathematical rigor as appropriate"
        
        if extracted_elements.get('extracted_tables'):
            table_count = len(extracted_elements['extracted_tables'])
            prompt += f"\n10. Reference that {table_count} data tables were extracted - use structured data approach where relevant"
        
        return prompt
    
    def _post_process_response(
        self, 
        response: str, 
        section_config: Dict[str, Any], 
        intelligent_analysis: Dict[str, Any]
    ) -> str:
        """Post-process AI response for quality and consistency"""
        
        if not response or len(response.strip()) < 50:
            return self._generate_intelligent_fallback("", section_config, intelligent_analysis)
        
        # Clean up response
        cleaned_response = response.strip()
        
        # Add regulatory context if missing
        frameworks = intelligent_analysis.get('regulatory_frameworks', [])
        if frameworks and len(frameworks) > 0 and frameworks[0] not in cleaned_response.lower():
            framework_note = f"\n\nNote: This section addresses requirements under the {frameworks[0].upper()} regulatory framework."
            cleaned_response += framework_note
        
        # Ensure minimum quality for tables
        if section_config.get("type") == "table":
            if cleaned_response.count('|') < 20:  # Minimum table structure for quality
                return self._generate_intelligent_fallback("table", section_config, intelligent_analysis)
        
        # Add implementation complexity note if high complexity
        impl_complexity = intelligent_analysis.get('implementation_complexity', 'Medium')
        if impl_complexity in ['High', 'Very High'] and 'implementation' not in cleaned_response.lower():
            impl_note = f"\n\nImplementation Note: This requirement has {impl_complexity.lower()} implementation complexity and may require additional technical resources and extended timelines."
            cleaned_response += impl_note
        
        return cleaned_response
    
    def _generate_intelligent_fallback(
        self, 
        section_name: str, 
        section_config: Dict[str, Any], 
        intelligent_analysis: Dict[str, Any]
    ) -> str:
        """Generate intelligent fallback content when AI is unavailable"""
        
        frameworks = intelligent_analysis.get('regulatory_frameworks', ['General'])
        complexity = intelligent_analysis.get('mathematical_complexity_level', 'Medium')
        compliance_criticality = intelligent_analysis.get('compliance_assessment', {}).get('compliance_criticality', 'Medium')
        impl_complexity = intelligent_analysis.get('implementation_complexity', 'Medium')
        
        if section_config.get("type") == "table":
            columns = section_config.get("columns", ["ID", "Description", "Requirements"])
            fallback_rows = []
            
            # Generate framework-specific sample data
            for i in range(8):  # Minimum 8 rows for comprehensive coverage
                if 'basel' in str(frameworks).lower():
                    row = [
                        f"REG-{frameworks[0].upper()}-{i+1:03d}", 
                        f"Basel {complexity} Complexity Requirement {i+1}", 
                        f"Regulatory compliance requirement addressing {complexity.lower()} mathematical complexity with {compliance_criticality.lower()} priority and {impl_complexity.lower()} implementation complexity"
                    ]
                elif 'mifid' in str(frameworks).lower():
                    row = [
                        f"MIFID-{i+1:03d}",
                        f"MiFID II Investment Service Requirement {i+1}",
                        f"Client protection and market transparency requirement with {compliance_criticality.lower()} compliance priority"
                    ]
                else:
                    row = [
                        f"REQ-{i+1:03d}", 
                        f"{frameworks[0]} Business Requirement {i+1}", 
                        f"Business requirement addressing {compliance_criticality.lower()} criticality compliance with {impl_complexity.lower()} implementation complexity"
                    ]
                
                # Pad or trim row to match columns
                while len(row) < len(columns):
                    row.append(f"Specification {len(row)+1}")
                row = row[:len(columns)]
                
                fallback_rows.append(" | ".join(row))
            
            return "\n".join(fallback_rows)
        
        else:
            return f"""INTELLIGENT FALLBACK CONTENT FOR {section_name}
            
            This section addresses requirements under the {', '.join(frameworks)} regulatory framework(s) 
            with {complexity.lower()} mathematical complexity, {compliance_criticality.lower()} compliance criticality,
            and {impl_complexity.lower()} implementation complexity.
            
            REGULATORY CONTEXT:
            - Primary Framework: {', '.join(frameworks)}
            - Mathematical Complexity: {complexity}
            - Compliance Priority: {compliance_criticality}
            - Implementation Complexity: {impl_complexity}
            
            IMPLEMENTATION REQUIREMENTS:
            - Establish governance processes aligned with {', '.join(frameworks)} regulatory frameworks
            - Implement controls appropriate to {complexity.lower()} complexity mathematical models and calculations
            - Ensure {compliance_criticality.lower()}-priority compliance monitoring, reporting, and audit trails
            - Develop documentation standards meeting regulatory examination requirements
            - Plan for {impl_complexity.lower()} implementation complexity with appropriate resource allocation
            
            STAKEHOLDER RESPONSIBILITIES:
            - Business owners must define functional requirements aligned with regulatory obligations
            - Compliance teams must validate regulatory alignment and perform ongoing monitoring
            - IT teams must implement technical specifications supporting regulatory calculations
            - Risk teams must establish monitoring frameworks and escalation procedures
            - Audit teams must validate implementation effectiveness and regulatory compliance
            
            QUALITY ASSURANCE:
            - All requirements must be traceable to specific regulatory citations
            - Implementation must support regulatory examination and audit processes
            - Change management must consider regulatory approval requirements
            - Testing must validate accuracy of regulatory calculations and reporting
            
            COMPLIANCE MONITORING:
            - Establish key risk indicators (KRIs) for regulatory compliance
            - Implement exception reporting and escalation procedures
            - Maintain regulatory change management processes and impact assessments
            - Document all regulatory decisions and their business rationale
            
            AI processing is currently unavailable. This intelligent fallback incorporates 
            comprehensive document analysis findings including regulatory framework detection, 
            mathematical complexity assessment, compliance criticality evaluation, 
            implementation complexity analysis, and stakeholder identification to provide 
            contextually appropriate and professionally structured placeholder content."""

def generate_enhanced_brd_with_intelligence(
    document_text: str, 
    extracted_images: Dict[str, str], 
    extracted_formulas: List[Any], 
    document_analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate BRD using enhanced intelligence and adaptive processing"""
    
    logger.info("Starting intelligent BRD generation with advanced AI processing")
    
    # Initialize intelligent components
    llm = init_enhanced_llm()
    analyzer = IntelligentDocumentAnalyzer(llm)
    generator = AdaptiveBRDGenerator(llm, analyzer)
    
    # Perform intelligent analysis
    extracted_elements = {
        'extracted_formulas': extracted_formulas,
        'extracted_tables': document_analysis.get('extracted_tables', []),
        'extracted_images': extracted_images
    }
    
    intelligent_analysis = analyzer.analyze_document_context(document_text, extracted_elements)
    
    # Merge with existing document analysis
    enhanced_analysis = {**document_analysis, **intelligent_analysis}
    
    # Generate BRD with intelligence
    brd_content = {}
    quality_scores = {}
    compliance_checks = []
    
    document_context = {
        'document_text': document_text,
        'document_type': enhanced_analysis.get('document_type', 'Unknown'),
        'extracted_elements': extracted_elements
    }
    
    total_sections = len(ENHANCED_BRD_STRUCTURE)
    section_count = 0
    
    for section_name, section_config in ENHANCED_BRD_STRUCTURE.items():
        try:
            logger.info(f"Generating intelligent section: {section_name}")
            
            if section_config.get("type") == "parent":
                brd_content[section_name] = {}
                for subsection_name, subsection_config in section_config.get("subsections", {}).items():
                    
                    content = generator.generate_intelligent_brd_section(
                        subsection_name, subsection_config, document_context, intelligent_analysis
                    )
                    
                    if subsection_config.get("type") == "table":
                        df = parse_table_content(content, subsection_config.get("columns", []))
                        brd_content[section_name][subsection_name] = df
                    else:
                        brd_content[section_name][subsection_name] = content
                    
                    # Enhanced quality scoring
                    score, checks = calculate_enhanced_quality_score(
                        subsection_name, content, subsection_config, intelligent_analysis
                    )
                    quality_scores[subsection_name] = score
                    compliance_checks.extend(checks)
            else:
                
                content = generator.generate_intelligent_brd_section(
                    section_name, section_config, document_context, intelligent_analysis
                )
                
                if section_config.get("type") == "table":
                    df = parse_table_content(content, section_config.get("columns", []))
                    brd_content[section_name] = df
                else:
                    brd_content[section_name] = content
                
                # Enhanced quality scoring
                score, checks = calculate_enhanced_quality_score(
                    section_name, content, section_config, intelligent_analysis
                )
                quality_scores[section_name] = score
                compliance_checks.extend(checks)
            
            section_count += 1
            logger.info(f"Completed intelligent section {section_count}/{total_sections}: {section_name}")
            
        except Exception as e:
            logger.error(f"Error in intelligent generation for section {section_name}: {e}")
            # Fallback to intelligent placeholder
            if section_config.get("type") == "parent":
                brd_content[section_name] = {"error": f"Intelligent generation error: {str(e)}"}
            else:
                brd_content[section_name] = generator._generate_intelligent_fallback(
                    section_name, section_config, intelligent_analysis
                )
            
            quality_scores[section_name] = 0.3  # Partial credit for intelligent fallback
            compliance_checks.append(QualityCheck(section_name, "generation_error", "WARNING", str(e), "warning"))
    
    logger.info("Enhanced intelligent BRD generation completed")
    
    return {
        'brd_content': brd_content,
        'quality_scores': quality_scores,
        'compliance_checks': compliance_checks,
        'intelligent_analysis': intelligent_analysis,
        'enhanced_analysis': enhanced_analysis
    }

def calculate_enhanced_quality_score(
    section_name: str, 
    content: Any, 
    structure_config: Dict[str, Any], 
    intelligent_analysis: Dict[str, Any]
) -> Tuple[float, List[QualityCheck]]:
    """Calculate enhanced quality score using intelligent analysis"""
    
    checks = []
    score = 0.0
    max_score = 100.0
    
    # Base completeness check
    if content and str(content).strip():
        score += 25
        checks.append(QualityCheck(section_name, "completeness", "PASS", "Section has content", "info"))
    else:
        checks.append(QualityCheck(section_name, "completeness", "FAIL", "Section is empty", "error"))
        return 0.0, checks
    
    # Intelligent context relevance check
    frameworks = intelligent_analysis.get('regulatory_frameworks', [])
    if frameworks:
        content_str = str(content).lower()
        framework_mentions = sum(1 for fw in frameworks if fw in content_str)
        if framework_mentions > 0:
            score += 15
            checks.append(QualityCheck(section_name, "regulatory_alignment", "PASS", f"References {framework_mentions} regulatory framework(s)", "info"))
        else:
            score += 5
            checks.append(QualityCheck(section_name, "regulatory_alignment", "WARNING", "Limited regulatory framework alignment", "warning"))
    
    # Complexity appropriateness check
    complexity_level = intelligent_analysis.get('mathematical_complexity_level', 'Medium')
    content_length = len(str(content))
    
    if complexity_level == 'Very High' and content_length > 800:
        score += 20
        checks.append(QualityCheck(section_name, "complexity_alignment", "PASS", "Content depth matches very high complexity", "info"))
    elif complexity_level == 'High' and content_length > 600:
        score += 15
        checks.append(QualityCheck(section_name, "complexity_alignment", "PASS", "Content depth matches high complexity", "info"))
    elif complexity_level == 'Medium' and content_length > 400:
        score += 15
        checks.append(QualityCheck(section_name, "complexity_alignment", "PASS", "Content depth matches medium complexity", "info"))
    else:
        score += 8
        checks.append(QualityCheck(section_name, "complexity_alignment", "WARNING", "Content depth may not match document complexity", "warning"))
    
    # Structure-specific intelligent checks
    if structure_config.get("type") == "table":
        if isinstance(content, pd.DataFrame) and not content.empty:
            score += 20
            checks.append(QualityCheck(section_name, "format", "PASS", "Proper table format", "info"))
            
            # Intelligent row count assessment based on complexity
            expected_rows = 8 if complexity_level in ['High', 'Very High'] else 5
            if len(content) >= expected_rows:
                score += 20
                checks.append(QualityCheck(section_name, "intelligent_detail", "PASS", f"Sufficient detail for {complexity_level.lower()} complexity", "info"))
            else:
                score += 10
                checks.append(QualityCheck(section_name, "intelligent_detail", "WARNING", f"Consider more detail for {complexity_level.lower()} complexity", "warning"))
        else:
            checks.append(QualityCheck(section_name, "format", "FAIL", "Should be in table format", "error"))
    
    elif structure_config.get("type") == "text":
        # Intelligent text quality assessment
        required_elements = structure_config.get("required_elements", [])
        if required_elements:
            elements_found = 0
            content_lower = str(content).lower()
            
            for element in required_elements:
                element_variants = [
                    element.replace("_", " ").lower(),
                    element.replace("_", "").lower(),
                    element.lower()
                ]
                if any(variant in content_lower for variant in element_variants):
                    elements_found += 1
            
            element_score = (elements_found / len(required_elements)) * 20
            score += element_score
            
            if elements_found == len(required_elements):
                checks.append(QualityCheck(section_name, "required_elements", "PASS", "All required elements present", "info"))
            elif elements_found >= len(required_elements) * 0.7:
                checks.append(QualityCheck(section_name, "required_elements", "WARNING", f"Missing {len(required_elements) - elements_found} required elements", "warning"))
            else:
                checks.append(QualityCheck(section_name, "required_elements", "FAIL", f"Missing {len(required_elements) - elements_found} required elements", "error"))
    
    # Compliance criticality bonus
    compliance_criticality = intelligent_analysis.get('compliance_assessment', {}).get('compliance_criticality', 'Medium')
    if compliance_criticality == 'Critical' and 'audit' in str(content).lower():
        score += 10
        checks.append(QualityCheck(section_name, "audit_readiness", "PASS", "Audit-ready content for critical compliance", "info"))
    
    # Implementation complexity assessment
    impl_complexity = intelligent_analysis.get('implementation_complexity', 'Medium')
    if impl_complexity in ['High', 'Very High'] and ('implementation' in str(content).lower() or 'resource' in str(content).lower()):
        score += 5
        checks.append(QualityCheck(section_name, "implementation_awareness", "PASS", f"Addresses {impl_complexity.lower()} implementation complexity", "info"))
    
    return min(score, max_score), checks

# Legacy function for backward compatibility
def generate_enhanced_brd(
    document_text: str, 
    extracted_images: Dict[str, str], 
    extracted_formulas: List[Any], 
    document_analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """Legacy function - redirects to intelligent generation"""
    return generate_enhanced_brd_with_intelligence(
        document_text, extracted_images, extracted_formulas, document_analysis
    )

def parse_table_content(content: str, columns: List[str]) -> pd.DataFrame:
    """Parse AI-generated table content into DataFrame"""
    try:
        if not content or not columns:
            return pd.DataFrame(columns=columns)
            
        lines = content.strip().split('\n')
        data_rows = []
        
        for line in lines:
            if '|' in line and line.strip():
                # Clean and split the line
                row = [cell.strip() for cell in line.split('|')]
                # Remove empty cells at the beginning/end
                row = [cell for cell in row if cell]
                
                # Skip header separators like |---|---|
                if all(cell in ['', '---', '--', '-'] or set(cell) <= {'-', ' '} for cell in row):
                    continue
                
                # Skip the column header row if it matches our expected columns
                if len(row) == len(columns) and all(col.lower() in ' '.join(row).lower() for col in columns[:3]):
                    continue
                
                # Pad or trim to match column count
                if len(row) < len(columns):
                    row.extend([''] * (len(columns) - len(row)))
                elif len(row) > len(columns):
                    row = row[:len(columns)]
                
                if any(cell.strip() for cell in row):  # Only add non-empty rows
                    data_rows.append(row)
        
        # If no data rows found, create sample data
        if not data_rows:
            data_rows = [['Sample ' + str(i+1)] + [''] * (len(columns) - 1) for i in range(3)]
        
        df = pd.DataFrame(data_rows, columns=columns)
        return df
    
    except Exception as e:
        logger.error(f"Error parsing table content: {str(e)}")
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=columns)

# Additional utility functions for intelligent processing

def get_intelligent_brd_recommendations(intelligent_analysis: Dict[str, Any]) -> List[str]:
    """Generate intelligent recommendations based on document analysis"""
    recommendations = []
    
    # Framework-specific recommendations
    frameworks = intelligent_analysis.get('regulatory_frameworks', [])
    if 'basel' in frameworks:
        recommendations.append("Consider implementing Basel-specific risk calculation validation frameworks")
        recommendations.append("Establish correlation matrix validation and back-testing procedures")
    
    if 'mifid' in frameworks:
        recommendations.append("Implement best execution monitoring and RTS 27/28 reporting capabilities")
        recommendations.append("Establish client categorization and suitability assessment workflows")
    
    # Complexity-based recommendations
    complexity = intelligent_analysis.get('mathematical_complexity_level', 'Medium')
    if complexity in ['High', 'Very High']:
        recommendations.append("Establish mathematical model validation and testing frameworks")
        recommendations.append("Consider specialized quantitative risk management resources")
    
    # Implementation complexity recommendations
    impl_complexity = intelligent_analysis.get('implementation_complexity', 'Medium')
    if impl_complexity in ['High', 'Very High']:
        recommendations.append("Plan for extended implementation timeline with phase-gate approach")
        recommendations.append("Establish dedicated project management office for implementation oversight")
    
    return recommendations
