/**
 * Sentinel Gmail Add-on - Server-side Apps Script
 * 
 * This script handles Gmail context and extracts email data when a message is opened.
 * It sends the data to the Sentinel backend for phishing detection analysis.
 */

// Configuration
const CONFIG = {
  // For local testing, use ngrok or similar to expose localhost
  // Replace with your actual backend URL
  BACKEND_URL: 'https://triumphal-nonempathically-tabetha.ngrok-free.dev/classify',
};

/**
 * Triggered when a user opens a Gmail message.
 * Returns a card to display in the side panel.
 * Automatically analyzes the email for phishing.
 * 
 * @param {Object} e - Event object containing Gmail message context
 * @return {Card} A card UI to display in the side panel
 */
function onGmailMessageOpen(e) {
  // Get the message ID from the event
  var messageId = e.gmail.messageId;
  var accessToken = e.gmail.accessToken;
  
  // Fetch email details
  var emailData = getEmailData(messageId, accessToken);
  
  // If there's an error getting email data, show email info card
  if (emailData.error) {
    return buildSidebarCard(emailData);
  }
  
  // Automatically analyze the email for phishing
  try {
    var analysisResult = performAnalysis(emailData.subject, emailData.from, emailData.rawBody);
    return buildAnalysisCard(emailData, analysisResult);
  } catch (error) {
    // If analysis fails, show error with retry button
    Logger.log('Auto-analysis error: ' + error.toString());
    return buildAnalysisErrorCard(emailData, error.toString());
  }
}

/**
 * Extracts email data using Gmail API
 * 
 * @param {string} messageId - The Gmail message ID
 * @param {string} accessToken - OAuth access token
 * @return {Object} Email data object with to, subject, from, body, and raw content
 */
function getEmailData(messageId, accessToken) {
  try {
    // Use Gmail API to get message details
    var message = GmailApp.getMessageById(messageId);
    
    if (!message) {
      return {
        subject: "Unable to load email",
        from: "Unknown",
        body: "Could not retrieve email content.",
        rawBody: "",
        error: true
      };
    }
    
    // Extract email components
    var subject = message.getSubject() || "(No Subject)";
    var from = message.getFrom() || "Unknown Sender";
    var to = message.getTo() || "Unknown Recipient";
    var plainBody = message.getPlainBody() || "";
    var htmlBody = message.getBody() || "";
    
    // Use plain body if available, otherwise strip HTML from body
    var bodyText = plainBody || stripHtmlTags(htmlBody);
    
    // Limit body length for display (first 1000 chars)
    var displayBody = bodyText.length > 1000 
      ? bodyText.substring(0, 1000) + "..." 
      : bodyText;
    
    return {
      to: to,
      subject: subject,
      from: from,
      body: displayBody,
      rawBody: bodyText, // Full body for API analysis
      error: false
    };
    
  } catch (error) {
    Logger.log("Error fetching email data: " + error.toString());
    return {
      to: "Unknown",
      subject: "Error",
      from: "Unknown",
      body: "Error: " + error.toString(),
      rawBody: "",
      error: true
    };
  }
}

/**
 * Strips HTML tags from a string
 * 
 * @param {string} html - HTML string
 * @return {string} Plain text
 */
function stripHtmlTags(html) {
  return html.replace(/<[^>]*>/g, ' ')
             .replace(/\s+/g, ' ')
             .trim();
}

/**
 * Builds the sidebar card UI (only used for errors now)
 * 
 * @param {Object} emailData - Email data object
 * @return {Card} Card UI for the sidebar
 */
function buildSidebarCard(emailData) {
  var card = CardService.newCardBuilder();
  
  // Header
  var header = CardService.newCardHeader()
    .setTitle('Gaurd')
    .setSubtitle('Phishing Detection');
  
  card.setHeader(header);
  
  // Email details section
  var section = CardService.newCardSection();
  
  // Subject widget
  section.addWidget(
    CardService.newKeyValue()
      .setTopLabel('Subject')
      .setContent(emailData.subject)
      .setMultiline(true)
  );

  // To widget
  section.addWidget(
    CardService.newKeyValue()
      .setTopLabel('To')
      .setContent(emailData.to)
      .setMultiline(true)
  );
  
  // From widget
  section.addWidget(
    CardService.newKeyValue()
      .setTopLabel('From')
      .setContent(emailData.from)
      .setMultiline(true)
  );
  
  // Divider
  section.addWidget(CardService.newDivider());
  
  // Body preview widget
  section.addWidget(
    CardService.newTextParagraph()
      .setText('<b>Email Body:</b>')
  );
  
  section.addWidget(
    CardService.newTextParagraph()
      .setText(emailData.body) // Show full display body (already limited to 1000 chars)
  );
  
  // Show error message if present
  if (emailData.error) {
    section.addWidget(CardService.newDivider());
    section.addWidget(
      CardService.newTextParagraph()
        .setText('⚠️ <b>Unable to load email for analysis</b>')
    );
  }
  
  card.addSection(section);
  
  return card.build();
}

/**
 * Performs phishing analysis by calling the backend API
 * 
 * @param {string} subject - Email subject
 * @param {string} from - Email sender  
 * @param {string} body - Email body
 * @return {Object} Analysis result from backend
 */
function performAnalysis(subject, from, body) {
  // Ensure BACKEND_URL is clean
  var backendUrl = CONFIG.BACKEND_URL.trim();
  Logger.log("Sending request to: " + backendUrl);
  Logger.log("Subject: " + subject);
  Logger.log("From: " + from);
  Logger.log("Body length: " + body.length);
  
  // Prepare the request payload
  var payload = {
    'text': body,
    'subject': subject
  };
  
  var options = {
    'method': 'post',
    'contentType': 'application/json',
    'payload': JSON.stringify(payload),
    'muteHttpExceptions': true,
    'timeout': 120  // Increased timeout for model loading
  };
  
  // Call the backend API
  var response = UrlFetchApp.fetch(backendUrl, options);
  var responseCode = response.getResponseCode();
  
  Logger.log("Response code: " + responseCode);
  
  if (responseCode === 200) {
    return JSON.parse(response.getContentText());
  } else {
    var errorText = response.getContentText();
    Logger.log("Backend error response: " + errorText);
    throw new Error('Backend returned error: ' + responseCode + '\\n' + errorText);
  }
}

/**
 * Builds a card showing the analysis results automatically
 * Uses Gemma-2-2b-it classification and explanation
 * 
 * @param {Object} emailData - Email data object
 * @param {Object} analysis - Analysis result from backend
 * @return {Card} Card with analysis results
 */
function buildAnalysisCard(emailData, analysis) {
  var card = CardService.newCardBuilder();
  
  // Header
  var isPhishing = analysis.label === 'phish';
  var header = CardService.newCardHeader();
  
  if (isPhishing) {
    header.setTitle('⚠️ PHISHING DETECTED')
          .setSubtitle('This email may be phishing');
  } else {
    header.setTitle('✅ SAFE')
          .setSubtitle('This email appears legitimate');
  }
  
  card.setHeader(header);
  
  // Email info section
  var emailSection = CardService.newCardSection();
  emailSection.addWidget(
    CardService.newKeyValue()
      .setTopLabel('Subject')
      .setContent(emailData.subject)
      .setMultiline(true)
  );
  emailSection.addWidget(
    CardService.newKeyValue()
      .setTopLabel('From')
      .setContent(emailData.from)
      .setMultiline(true)
  );
  card.addSection(emailSection);
  
  // Analysis section
  var analysisSection = CardService.newCardSection();
  
  // Show confidence
  var confidence = (analysis.confidence * 100).toFixed(0) + '%';
  
  analysisSection.addWidget(
    CardService.newKeyValue()
      .setTopLabel('Confidence')
      .setContent(confidence)
  );
  
  analysisSection.addWidget(CardService.newDivider());
  
  // Explanation - parse and clean up
  if (analysis.explanation) {
    var cleanExplanation = parseGemmaExplanation(analysis.explanation);
    analysisSection.addWidget(
      CardService.newTextParagraph()
        .setText('<b>📋 Analysis:</b><br><br>' + cleanExplanation)
    );
  }
  
  card.addSection(analysisSection);
  
  return card.build();
}

/**
 * Builds an error card with retry button
 * 
 * @param {Object} emailData - Email data object
 * @param {string} errorMessage - Error message
 * @return {Card} Error card with retry option
 */
function buildAnalysisErrorCard(emailData, errorMessage) {
  var card = CardService.newCardBuilder();
  
  var header = CardService.newCardHeader()
    .setTitle('⚠️ Gaurd')
    .setSubtitle('Analysis Failed');
  
  card.setHeader(header);
  
  // Email info section
  var emailSection = CardService.newCardSection();
  emailSection.addWidget(
    CardService.newKeyValue()
      .setTopLabel('Subject')
      .setContent(emailData.subject)
      .setMultiline(true)
  );
  emailSection.addWidget(
    CardService.newKeyValue()
      .setTopLabel('From')
      .setContent(emailData.from)
      .setMultiline(true)
  );
  card.addSection(emailSection);
  
  // Error section
  var errorSection = CardService.newCardSection();
  
  // Format error message
  var displayError = errorMessage;
  if (errorMessage.indexOf('timed out') > -1 || errorMessage.indexOf('timeout') > -1) {
    displayError = '⏱️ Analysis took too long (timeout).\\n\\n⚠️ This usually means the backend is still loading models.\\n\\n✅ Solution:\\n1. Wait 30 seconds\\n2. Click the retry button below\\n3. The models should be loaded by then\\n\\nNote: First request after server restart takes 30-60 seconds for model loading.';
  } else if (errorMessage.indexOf('connect') > -1 || errorMessage.indexOf('fetch') > -1) {
    displayError = '🔌 Unable to connect to backend.\\n\\nMake sure:\\n1. Backend is running on port 8000\\n2. ngrok is active for localhost exposure\\n3. Internet connection is stable\\n\\nError: ' + errorMessage;
  }
  
  errorSection.addWidget(
    CardService.newTextParagraph()
      .setText(displayError)
  );
  
  errorSection.addWidget(CardService.newDivider());
  
  // Add retry button
  var retryButton = CardService.newTextButton()
    .setText('🔄 Retry Analysis')
    .setOnClickAction(
      CardService.newAction()
        .setFunctionName('analyzeEmail')
        .setParameters({
          'subject': emailData.subject,
          'from': emailData.from,
          'body': emailData.rawBody
        })
    );
  
  errorSection.addWidget(
    CardService.newButtonSet()
      .addButton(retryButton)
  );
  
  card.addSection(errorSection);
  
  return card.build();
}

/**
 * Analyzes the email using the Sentinel backend API
 * This function is used by the retry button when auto-analysis fails
 * 
 * @param {Object} e - Event object containing email data as parameters
 * @return {ActionResponse} Updated card with analysis results
 */
function analyzeEmail(e) {
  var params = e.parameters;
  var subject = (params.subject || '').trim();
  var from = (params.from || '').trim();
  var body = (params.body || '').trim();
  
  try {
    var analysisResult = performAnalysis(subject, from, body);
    
    // Create email data structure for card building
    var emailData = {
      subject: subject,
      from: from,
      to: '',  // Not needed for display in retry
      body: body.substring(0, 1000),
      rawBody: body,
      error: false
    };
    
    var card = buildAnalysisCard(emailData, analysisResult);
    var navigation = CardService.newNavigation().updateCard(card);
    return CardService.newActionResponseBuilder()
      .setNavigation(navigation)
      .build();
      
  } catch (error) {
    Logger.log('Retry analysis error: ' + error.toString());
    
    // Create email data structure for error card
    var emailData = {
      subject: subject,
      from: from,
      to: '',
      body: body.substring(0, 1000),
      rawBody: body,
      error: false
    };
    
    var errorCard = buildAnalysisErrorCard(emailData, error.toString());
    var navigation = CardService.newNavigation().updateCard(errorCard);
    return CardService.newActionResponseBuilder()
      .setNavigation(navigation)
      .build();
  }
}

/**
 * Parses Gemma's JSON output and extracts just the explanation text
 * If JSON parsing fails, cleans up the raw text
 * 
 * @param {string} text - Raw output from Gemma (possibly JSON)
 * @return {string} Cleaned explanation text
 */
function parseGemmaExplanation(text) {
  if (!text) return 'No explanation available.';
  
  // First, check if the text is already formatted (not JSON)
  // The backend may return pre-formatted text when JSON parsing fails
  if (text.indexOf('Classification:') === 0 || text.indexOf('⚠️') === 0) {
    // Already formatted by backend, use as-is
    return cleanMarkdown(text);
  }
  
  try {
    // Try to parse as JSON (Gemma was fine-tuned to output JSON)
    var startIdx = text.indexOf('{');
    var endIdx = text.lastIndexOf('}');
    
    if (startIdx !== -1 && endIdx !== -1) {
      var jsonStr = text.substring(startIdx, endIdx + 1);
      var parsed = JSON.parse(jsonStr);
      
      // Extract all fields and format properly
      var sections = [];
      
      // Classification
      if (parsed.label) {
        sections.push('Classification: ' + parsed.label);
      }
      
      // Explanation
      if (parsed.explanation) {
        sections.push('\nExplanation:\n' + cleanMarkdown(parsed.explanation));
      }
      
      // Evidence snippets (limit to first 2)
      if (parsed.evidence_snippets && Array.isArray(parsed.evidence_snippets)) {
        var evidence = parsed.evidence_snippets
          .filter(function(s) { return s; })
          .slice(0, 2)  // Limit to first 2 items
          .map(function(s) { return '• ' + s; })
          .join('\n');
        if (evidence) {
          sections.push('\nEvidence:\n' + evidence);
        }
      }
      
      // User advice (limit to first 2)
      if (parsed.user_advice && Array.isArray(parsed.user_advice)) {
        var advice = parsed.user_advice
          .filter(function(a) { return a; })
          .slice(0, 2)  // Limit to first 2 items
          .map(function(a) { return '• ' + a; })
          .join('\n');
        if (advice) {
          sections.push('\nUser Advice:\n' + advice);
        }
      }
      
      if (sections.length > 0) {
        return sections.join('\n');
      }
    }
  } catch (e) {
    // JSON parsing failed, use raw text
    Logger.log('Could not parse Gemma output as JSON: ' + e.toString());
  }
  
  // Fallback: clean the raw text
  return cleanMarkdown(text);
}

/**
 * Cleans up markdown formatting from text
 * Removes markdown characters like ##, **, etc.
 * 
 * @param {string} text - Text with markdown
 * @return {string} Cleaned text
 */
function cleanMarkdown(text) {
  if (!text) return '';
  
  // Remove markdown headings (##, ###, etc.)
  text = text.replace(/^#+\s+/gm, '');
  
  // Remove bold markdown (**)
  text = text.replace(/\*\*([^*]+)\*\*/g, '$1');
  
  // Remove italic markdown (*)
  text = text.replace(/\*([^*]+)\*/g, '$1');
  
  // Clean up multiple consecutive newlines
  text = text.replace(/\n\n+/g, '\n\n');
  
  // Remove leading/trailing whitespace
  text = text.trim();
  
  return text;
}
