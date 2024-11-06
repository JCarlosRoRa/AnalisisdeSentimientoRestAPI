$(document).ready(function() {
  $('#sentiment-form').on('submit', function(e) {
      e.preventDefault();

      const text = $('#text-input').val();
      
      $.ajax({
          type: "POST",
          url: "/api/sentiment/",
          contentType: "application/json",
          data: JSON.stringify({ text: text }),
          success: function(data) {
              // Mostrar los resultados en el HTML
              $('#bert-label').text(data.bert_sentiment.label);
              $('#bert-confidence').text(data.bert_sentiment.confidence.toFixed(2)); // Mostrar la confianza con 2 decimales
              
              // Mostrar polaridad y subjetividad
              $('#polarity').text(data.textblob_sentiment.polarity.toFixed(2));      // Redondeado a 2 decimales
              $('#subjectivity').text(data.textblob_sentiment.subjectivity.toFixed(2)); // Redondeado a 2 decimales
              
              $('#result').show();
          },
          error: function() {
              alert("Error al analizar el sentimiento.");
          }
      });
  });
});
